#define EASYSIMD_TEST_X86_AVX512_INSN roundscale

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/roundscale.h>
#include <easysimd/x86/avx512/setzero.h>

static int
test_easysimd_mm_roundscale_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const int32_t imm8;
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   184.58), EASYSIMD_FLOAT32_C(  -477.45), EASYSIMD_FLOAT32_C(  -816.99), EASYSIMD_FLOAT32_C(  -969.27) },
       INT32_C(           0),
      { EASYSIMD_FLOAT32_C(   185.00), EASYSIMD_FLOAT32_C(  -477.00), EASYSIMD_FLOAT32_C(  -817.00), EASYSIMD_FLOAT32_C(  -969.00) } },
    { { EASYSIMD_FLOAT32_C(  -630.66), EASYSIMD_FLOAT32_C(  -650.78), EASYSIMD_FLOAT32_C(   424.73), EASYSIMD_FLOAT32_C(  -953.66) },
       INT32_C(          16),
      { EASYSIMD_FLOAT32_C(  -630.50), EASYSIMD_FLOAT32_C(  -651.00), EASYSIMD_FLOAT32_C(   424.50), EASYSIMD_FLOAT32_C(  -953.50) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    49.10), EASYSIMD_FLOAT32_C(   398.66), EASYSIMD_FLOAT32_C(   620.43) },
       INT32_C(          32),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    49.00), EASYSIMD_FLOAT32_C(   398.75), EASYSIMD_FLOAT32_C(   620.50) } },
    { { EASYSIMD_FLOAT32_C(   -29.44), EASYSIMD_FLOAT32_C(  -399.88), EASYSIMD_FLOAT32_C(    -8.17), EASYSIMD_FLOAT32_C(  -624.68) },
       INT32_C(          48),
      { EASYSIMD_FLOAT32_C(   -29.50), EASYSIMD_FLOAT32_C(  -399.88), EASYSIMD_FLOAT32_C(    -8.12), EASYSIMD_FLOAT32_C(  -624.62) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   762.14), EASYSIMD_FLOAT32_C(   463.35), EASYSIMD_FLOAT32_C(  -105.13) },
       INT32_C(          64),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   762.12), EASYSIMD_FLOAT32_C(   463.38), EASYSIMD_FLOAT32_C(  -105.12) } },
    { { EASYSIMD_FLOAT32_C(  -338.82), EASYSIMD_FLOAT32_C(  -677.32), EASYSIMD_FLOAT32_C(  -711.13), EASYSIMD_FLOAT32_C(    39.83) },
       INT32_C(          80),
      { EASYSIMD_FLOAT32_C(  -338.81), EASYSIMD_FLOAT32_C(  -677.31), EASYSIMD_FLOAT32_C(  -711.12), EASYSIMD_FLOAT32_C(    39.84) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   222.84), EASYSIMD_FLOAT32_C(   537.98), EASYSIMD_FLOAT32_C(   127.16) },
       INT32_C(          96),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   222.84), EASYSIMD_FLOAT32_C(   537.98), EASYSIMD_FLOAT32_C(   127.16) } },
    { { EASYSIMD_FLOAT32_C(  -448.10),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -985.10), EASYSIMD_FLOAT32_C(   -93.24) },
       INT32_C(         112),
      { EASYSIMD_FLOAT32_C(  -448.10),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -985.10), EASYSIMD_FLOAT32_C(   -93.24) } },
    { { EASYSIMD_FLOAT32_C(   426.07),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   497.75), EASYSIMD_FLOAT32_C(  -973.80) },
       INT32_C(         128),
      { EASYSIMD_FLOAT32_C(   426.07),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   497.75), EASYSIMD_FLOAT32_C(  -973.80) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   635.20), EASYSIMD_FLOAT32_C(   314.85), EASYSIMD_FLOAT32_C(   453.80) },
       INT32_C(         144),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   635.20), EASYSIMD_FLOAT32_C(   314.85), EASYSIMD_FLOAT32_C(   453.80) } },
    { { EASYSIMD_FLOAT32_C(  -885.03), EASYSIMD_FLOAT32_C(   407.49), EASYSIMD_FLOAT32_C(  -605.40), EASYSIMD_FLOAT32_C(   154.81) },
       INT32_C(         160),
      { EASYSIMD_FLOAT32_C(  -885.03), EASYSIMD_FLOAT32_C(   407.49), EASYSIMD_FLOAT32_C(  -605.40), EASYSIMD_FLOAT32_C(   154.81) } },
    { { EASYSIMD_FLOAT32_C(   206.02),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -547.27), EASYSIMD_FLOAT32_C(  -666.82) },
       INT32_C(         176),
      { EASYSIMD_FLOAT32_C(   206.02),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -547.27), EASYSIMD_FLOAT32_C(  -666.82) } },
    { { EASYSIMD_FLOAT32_C(   608.35),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   791.84), EASYSIMD_FLOAT32_C(  -704.03) },
       INT32_C(         192),
      { EASYSIMD_FLOAT32_C(   608.35),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   791.84), EASYSIMD_FLOAT32_C(  -704.03) } },
    { { EASYSIMD_FLOAT32_C(   413.65), EASYSIMD_FLOAT32_C(   816.78), EASYSIMD_FLOAT32_C(   748.24), EASYSIMD_FLOAT32_C(  -949.28) },
       INT32_C(         208),
      { EASYSIMD_FLOAT32_C(   413.65), EASYSIMD_FLOAT32_C(   816.78), EASYSIMD_FLOAT32_C(   748.24), EASYSIMD_FLOAT32_C(  -949.28) } },
    { { EASYSIMD_FLOAT32_C(   599.73), EASYSIMD_FLOAT32_C(  -390.36), EASYSIMD_FLOAT32_C(   325.04), EASYSIMD_FLOAT32_C(   -85.42) },
       INT32_C(         224),
      { EASYSIMD_FLOAT32_C(   599.73), EASYSIMD_FLOAT32_C(  -390.36), EASYSIMD_FLOAT32_C(   325.04), EASYSIMD_FLOAT32_C(   -85.42) } },
    { { EASYSIMD_FLOAT32_C(  -590.15),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -821.59), EASYSIMD_FLOAT32_C(   817.34) },
       INT32_C(         240),
      { EASYSIMD_FLOAT32_C(  -590.15),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -821.59), EASYSIMD_FLOAT32_C(   817.34) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   710.86), EASYSIMD_FLOAT32_C(   184.82), EASYSIMD_FLOAT32_C(   -45.89) },
       INT32_C(           1),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   710.00), EASYSIMD_FLOAT32_C(   184.00), EASYSIMD_FLOAT32_C(   -46.00) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   289.03), EASYSIMD_FLOAT32_C(   879.59), EASYSIMD_FLOAT32_C(   631.03) },
       INT32_C(          17),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   289.00), EASYSIMD_FLOAT32_C(   879.50), EASYSIMD_FLOAT32_C(   631.00) } },
    { { EASYSIMD_FLOAT32_C(   950.06), EASYSIMD_FLOAT32_C(   307.04), EASYSIMD_FLOAT32_C(    61.64), EASYSIMD_FLOAT32_C(   766.84) },
       INT32_C(          33),
      { EASYSIMD_FLOAT32_C(   950.00), EASYSIMD_FLOAT32_C(   307.00), EASYSIMD_FLOAT32_C(    61.50), EASYSIMD_FLOAT32_C(   766.75) } },
    { { EASYSIMD_FLOAT32_C(   112.35),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -344.99), EASYSIMD_FLOAT32_C(   721.99) },
       INT32_C(          49),
      { EASYSIMD_FLOAT32_C(   112.25),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -345.00), EASYSIMD_FLOAT32_C(   721.88) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -410.10), EASYSIMD_FLOAT32_C(   963.84), EASYSIMD_FLOAT32_C(     8.91) },
       INT32_C(          65),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -410.12), EASYSIMD_FLOAT32_C(   963.81), EASYSIMD_FLOAT32_C(     8.88) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -374.27), EASYSIMD_FLOAT32_C(     7.92), EASYSIMD_FLOAT32_C(   -74.18) },
       INT32_C(          81),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -374.28), EASYSIMD_FLOAT32_C(     7.91), EASYSIMD_FLOAT32_C(   -74.19) } },
    { { EASYSIMD_FLOAT32_C(  -549.43), EASYSIMD_FLOAT32_C(   419.03), EASYSIMD_FLOAT32_C(   977.64), EASYSIMD_FLOAT32_C(  -669.85) },
       INT32_C(          97),
      { EASYSIMD_FLOAT32_C(  -549.44), EASYSIMD_FLOAT32_C(   419.02), EASYSIMD_FLOAT32_C(   977.62), EASYSIMD_FLOAT32_C(  -669.86) } },
    { { EASYSIMD_FLOAT32_C(   562.65), EASYSIMD_FLOAT32_C(   978.14), EASYSIMD_FLOAT32_C(     0.12), EASYSIMD_FLOAT32_C(  -130.31) },
       INT32_C(         113),
      { EASYSIMD_FLOAT32_C(   562.65), EASYSIMD_FLOAT32_C(   978.13), EASYSIMD_FLOAT32_C(     0.12), EASYSIMD_FLOAT32_C(  -130.31) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   924.97), EASYSIMD_FLOAT32_C(  -847.87), EASYSIMD_FLOAT32_C(  -776.36) },
       INT32_C(         129),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   924.97), EASYSIMD_FLOAT32_C(  -847.87), EASYSIMD_FLOAT32_C(  -776.36) } },
    { { EASYSIMD_FLOAT32_C(     5.36),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   659.56), EASYSIMD_FLOAT32_C(  -803.07) },
       INT32_C(         145),
      { EASYSIMD_FLOAT32_C(     5.36),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   659.56), EASYSIMD_FLOAT32_C(  -803.07) } },
    { { EASYSIMD_FLOAT32_C(  -255.70), EASYSIMD_FLOAT32_C(   -79.54), EASYSIMD_FLOAT32_C(   -53.15), EASYSIMD_FLOAT32_C(   370.03) },
       INT32_C(         161),
      { EASYSIMD_FLOAT32_C(  -255.70), EASYSIMD_FLOAT32_C(   -79.54), EASYSIMD_FLOAT32_C(   -53.15), EASYSIMD_FLOAT32_C(   370.03) } },
    { { EASYSIMD_FLOAT32_C(   872.67), EASYSIMD_FLOAT32_C(   -50.13), EASYSIMD_FLOAT32_C(  -383.01), EASYSIMD_FLOAT32_C(  -676.76) },
       INT32_C(         177),
      { EASYSIMD_FLOAT32_C(   872.67), EASYSIMD_FLOAT32_C(   -50.13), EASYSIMD_FLOAT32_C(  -383.01), EASYSIMD_FLOAT32_C(  -676.76) } },
    { { EASYSIMD_FLOAT32_C(  -405.36),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   418.96), EASYSIMD_FLOAT32_C(  -842.72) },
       INT32_C(         193),
      { EASYSIMD_FLOAT32_C(  -405.36),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   418.96), EASYSIMD_FLOAT32_C(  -842.72) } },
    { { EASYSIMD_FLOAT32_C(   671.31), EASYSIMD_FLOAT32_C(   186.05), EASYSIMD_FLOAT32_C(   -48.06), EASYSIMD_FLOAT32_C(   823.45) },
       INT32_C(         209),
      { EASYSIMD_FLOAT32_C(   671.31), EASYSIMD_FLOAT32_C(   186.05), EASYSIMD_FLOAT32_C(   -48.06), EASYSIMD_FLOAT32_C(   823.45) } },
    { { EASYSIMD_FLOAT32_C(   531.93), EASYSIMD_FLOAT32_C(   697.57), EASYSIMD_FLOAT32_C(  -584.95), EASYSIMD_FLOAT32_C(   681.51) },
       INT32_C(         225),
      { EASYSIMD_FLOAT32_C(   531.93), EASYSIMD_FLOAT32_C(   697.57), EASYSIMD_FLOAT32_C(  -584.95), EASYSIMD_FLOAT32_C(   681.51) } },
    { { EASYSIMD_FLOAT32_C(  -388.02), EASYSIMD_FLOAT32_C(  -579.00), EASYSIMD_FLOAT32_C(   -19.47), EASYSIMD_FLOAT32_C(   817.83) },
       INT32_C(         241),
      { EASYSIMD_FLOAT32_C(  -388.02), EASYSIMD_FLOAT32_C(  -579.00), EASYSIMD_FLOAT32_C(   -19.47), EASYSIMD_FLOAT32_C(   817.83) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -235.32), EASYSIMD_FLOAT32_C(  -464.67), EASYSIMD_FLOAT32_C(   829.38) },
       INT32_C(           2),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -235.00), EASYSIMD_FLOAT32_C(  -464.00), EASYSIMD_FLOAT32_C(   830.00) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   -39.42), EASYSIMD_FLOAT32_C(   854.11), EASYSIMD_FLOAT32_C(    41.00) },
       INT32_C(          18),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   -39.00), EASYSIMD_FLOAT32_C(   854.50), EASYSIMD_FLOAT32_C(    41.00) } },
    { { EASYSIMD_FLOAT32_C(   198.28), EASYSIMD_FLOAT32_C(  -754.49), EASYSIMD_FLOAT32_C(   692.15), EASYSIMD_FLOAT32_C(  -774.75) },
       INT32_C(          34),
      { EASYSIMD_FLOAT32_C(   198.50), EASYSIMD_FLOAT32_C(  -754.25), EASYSIMD_FLOAT32_C(   692.25), EASYSIMD_FLOAT32_C(  -774.75) } },
    { { EASYSIMD_FLOAT32_C(  -121.80), EASYSIMD_FLOAT32_C(   177.20), EASYSIMD_FLOAT32_C(   740.27), EASYSIMD_FLOAT32_C(  -712.11) },
       INT32_C(          50),
      { EASYSIMD_FLOAT32_C(  -121.75), EASYSIMD_FLOAT32_C(   177.25), EASYSIMD_FLOAT32_C(   740.38), EASYSIMD_FLOAT32_C(  -712.00) } },
    { { EASYSIMD_FLOAT32_C(   437.85), EASYSIMD_FLOAT32_C(  -297.06), EASYSIMD_FLOAT32_C(  -609.36), EASYSIMD_FLOAT32_C(  -205.02) },
       INT32_C(          66),
      { EASYSIMD_FLOAT32_C(   437.88), EASYSIMD_FLOAT32_C(  -297.00), EASYSIMD_FLOAT32_C(  -609.31), EASYSIMD_FLOAT32_C(  -205.00) } },
    { { EASYSIMD_FLOAT32_C(  -188.36), EASYSIMD_FLOAT32_C(   775.51), EASYSIMD_FLOAT32_C(   132.74), EASYSIMD_FLOAT32_C(   976.93) },
       INT32_C(          82),
      { EASYSIMD_FLOAT32_C(  -188.34), EASYSIMD_FLOAT32_C(   775.53), EASYSIMD_FLOAT32_C(   132.75), EASYSIMD_FLOAT32_C(   976.94) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -487.74), EASYSIMD_FLOAT32_C(   505.88), EASYSIMD_FLOAT32_C(  -465.24) },
       INT32_C(          98),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -487.73), EASYSIMD_FLOAT32_C(   505.89), EASYSIMD_FLOAT32_C(  -465.23) } },
    { { EASYSIMD_FLOAT32_C(   495.34), EASYSIMD_FLOAT32_C(   851.57), EASYSIMD_FLOAT32_C(    -6.75), EASYSIMD_FLOAT32_C(   109.32) },
       INT32_C(         114),
      { EASYSIMD_FLOAT32_C(   495.34), EASYSIMD_FLOAT32_C(   851.57), EASYSIMD_FLOAT32_C(    -6.75), EASYSIMD_FLOAT32_C(   109.32) } },
    { { EASYSIMD_FLOAT32_C(  -808.46), EASYSIMD_FLOAT32_C(   354.82), EASYSIMD_FLOAT32_C(  -183.20), EASYSIMD_FLOAT32_C(  -583.21) },
       INT32_C(         130),
      { EASYSIMD_FLOAT32_C(  -808.46), EASYSIMD_FLOAT32_C(   354.82), EASYSIMD_FLOAT32_C(  -183.20), EASYSIMD_FLOAT32_C(  -583.21) } },
    { { EASYSIMD_FLOAT32_C(   695.00), EASYSIMD_FLOAT32_C(   593.99), EASYSIMD_FLOAT32_C(    11.92), EASYSIMD_FLOAT32_C(   982.89) },
       INT32_C(         146),
      { EASYSIMD_FLOAT32_C(   695.00), EASYSIMD_FLOAT32_C(   593.99), EASYSIMD_FLOAT32_C(    11.92), EASYSIMD_FLOAT32_C(   982.89) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -314.18), EASYSIMD_FLOAT32_C(  -306.25), EASYSIMD_FLOAT32_C(   244.74) },
       INT32_C(         162),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -314.18), EASYSIMD_FLOAT32_C(  -306.25), EASYSIMD_FLOAT32_C(   244.74) } },
    { { EASYSIMD_FLOAT32_C(    20.26), EASYSIMD_FLOAT32_C(   133.48), EASYSIMD_FLOAT32_C(   482.32), EASYSIMD_FLOAT32_C(  -303.23) },
       INT32_C(         178),
      { EASYSIMD_FLOAT32_C(    20.26), EASYSIMD_FLOAT32_C(   133.48), EASYSIMD_FLOAT32_C(   482.32), EASYSIMD_FLOAT32_C(  -303.23) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -797.35), EASYSIMD_FLOAT32_C(   565.65), EASYSIMD_FLOAT32_C(   992.05) },
       INT32_C(         194),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -797.35), EASYSIMD_FLOAT32_C(   565.65), EASYSIMD_FLOAT32_C(   992.05) } },
    { { EASYSIMD_FLOAT32_C(   843.62), EASYSIMD_FLOAT32_C(   148.16), EASYSIMD_FLOAT32_C(  -829.69), EASYSIMD_FLOAT32_C(   -31.74) },
       INT32_C(         210),
      { EASYSIMD_FLOAT32_C(   843.62), EASYSIMD_FLOAT32_C(   148.16), EASYSIMD_FLOAT32_C(  -829.69), EASYSIMD_FLOAT32_C(   -31.74) } },
    { { EASYSIMD_FLOAT32_C(   525.13),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   756.48), EASYSIMD_FLOAT32_C(  -203.22) },
       INT32_C(         226),
      { EASYSIMD_FLOAT32_C(   525.13),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   756.48), EASYSIMD_FLOAT32_C(  -203.22) } },
    { { EASYSIMD_FLOAT32_C(   462.95), EASYSIMD_FLOAT32_C(   653.59), EASYSIMD_FLOAT32_C(  -741.53), EASYSIMD_FLOAT32_C(  -851.23) },
       INT32_C(         242),
      { EASYSIMD_FLOAT32_C(   462.95), EASYSIMD_FLOAT32_C(   653.59), EASYSIMD_FLOAT32_C(  -741.53), EASYSIMD_FLOAT32_C(  -851.23) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -850.49), EASYSIMD_FLOAT32_C(   852.73), EASYSIMD_FLOAT32_C(  -476.53) },
       INT32_C(           3),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -850.00), EASYSIMD_FLOAT32_C(   852.00), EASYSIMD_FLOAT32_C(  -476.00) } },
    { { EASYSIMD_FLOAT32_C(   220.24), EASYSIMD_FLOAT32_C(  -686.13), EASYSIMD_FLOAT32_C(   329.63), EASYSIMD_FLOAT32_C(   422.89) },
       INT32_C(          19),
      { EASYSIMD_FLOAT32_C(   220.00), EASYSIMD_FLOAT32_C(  -686.00), EASYSIMD_FLOAT32_C(   329.50), EASYSIMD_FLOAT32_C(   422.50) } },
    { { EASYSIMD_FLOAT32_C(   321.68), EASYSIMD_FLOAT32_C(   577.80), EASYSIMD_FLOAT32_C(   -59.48), EASYSIMD_FLOAT32_C(   165.30) },
       INT32_C(          35),
      { EASYSIMD_FLOAT32_C(   321.50), EASYSIMD_FLOAT32_C(   577.75), EASYSIMD_FLOAT32_C(   -59.25), EASYSIMD_FLOAT32_C(   165.25) } },
    { { EASYSIMD_FLOAT32_C(   110.83), EASYSIMD_FLOAT32_C(  -866.44), EASYSIMD_FLOAT32_C(  -934.35), EASYSIMD_FLOAT32_C(  -364.04) },
       INT32_C(          51),
      { EASYSIMD_FLOAT32_C(   110.75), EASYSIMD_FLOAT32_C(  -866.38), EASYSIMD_FLOAT32_C(  -934.25), EASYSIMD_FLOAT32_C(  -364.00) } },
    { { EASYSIMD_FLOAT32_C(   822.13), EASYSIMD_FLOAT32_C(   432.74), EASYSIMD_FLOAT32_C(   398.69), EASYSIMD_FLOAT32_C(   172.60) },
       INT32_C(          67),
      { EASYSIMD_FLOAT32_C(   822.12), EASYSIMD_FLOAT32_C(   432.69), EASYSIMD_FLOAT32_C(   398.69), EASYSIMD_FLOAT32_C(   172.56) } },
    { { EASYSIMD_FLOAT32_C(  -138.37),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   499.91), EASYSIMD_FLOAT32_C(    10.40) },
       INT32_C(          83),
      { EASYSIMD_FLOAT32_C(  -138.34),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   499.91), EASYSIMD_FLOAT32_C(    10.38) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   526.59), EASYSIMD_FLOAT32_C(  -557.11), EASYSIMD_FLOAT32_C(  -638.70) },
       INT32_C(          99),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   526.58), EASYSIMD_FLOAT32_C(  -557.11), EASYSIMD_FLOAT32_C(  -638.69) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -830.28), EASYSIMD_FLOAT32_C(  -363.71), EASYSIMD_FLOAT32_C(    12.61) },
       INT32_C(         115),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -830.27), EASYSIMD_FLOAT32_C(  -363.70), EASYSIMD_FLOAT32_C(    12.61) } },
    { { EASYSIMD_FLOAT32_C(  -822.09),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -312.37), EASYSIMD_FLOAT32_C(  -688.53) },
       INT32_C(         131),
      { EASYSIMD_FLOAT32_C(  -822.09),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -312.37), EASYSIMD_FLOAT32_C(  -688.53) } },
    { { EASYSIMD_FLOAT32_C(  -638.75),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   628.78), EASYSIMD_FLOAT32_C(   533.85) },
       INT32_C(         147),
      { EASYSIMD_FLOAT32_C(  -638.75),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   628.78), EASYSIMD_FLOAT32_C(   533.85) } },
    { { EASYSIMD_FLOAT32_C(   497.68), EASYSIMD_FLOAT32_C(   500.82), EASYSIMD_FLOAT32_C(   533.57), EASYSIMD_FLOAT32_C(  -499.20) },
       INT32_C(         163),
      { EASYSIMD_FLOAT32_C(   497.68), EASYSIMD_FLOAT32_C(   500.82), EASYSIMD_FLOAT32_C(   533.57), EASYSIMD_FLOAT32_C(  -499.20) } },
    { { EASYSIMD_FLOAT32_C(  -440.17), EASYSIMD_FLOAT32_C(  -972.62), EASYSIMD_FLOAT32_C(   103.62), EASYSIMD_FLOAT32_C(   -78.87) },
       INT32_C(         179),
      { EASYSIMD_FLOAT32_C(  -440.17), EASYSIMD_FLOAT32_C(  -972.62), EASYSIMD_FLOAT32_C(   103.62), EASYSIMD_FLOAT32_C(   -78.87) } },
    { { EASYSIMD_FLOAT32_C(   860.38),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   -56.08), EASYSIMD_FLOAT32_C(  -503.33) },
       INT32_C(         195),
      { EASYSIMD_FLOAT32_C(   860.38),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   -56.08), EASYSIMD_FLOAT32_C(  -503.33) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   164.90), EASYSIMD_FLOAT32_C(  -238.89), EASYSIMD_FLOAT32_C(  -885.95) },
       INT32_C(         211),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   164.90), EASYSIMD_FLOAT32_C(  -238.89), EASYSIMD_FLOAT32_C(  -885.95) } },
    { { EASYSIMD_FLOAT32_C(  -655.85), EASYSIMD_FLOAT32_C(  -934.74), EASYSIMD_FLOAT32_C(  -158.97), EASYSIMD_FLOAT32_C(   972.94) },
       INT32_C(         227),
      { EASYSIMD_FLOAT32_C(  -655.85), EASYSIMD_FLOAT32_C(  -934.74), EASYSIMD_FLOAT32_C(  -158.97), EASYSIMD_FLOAT32_C(   972.94) } },
    { { EASYSIMD_FLOAT32_C(  -161.19), EASYSIMD_FLOAT32_C(  -536.65), EASYSIMD_FLOAT32_C(   959.15), EASYSIMD_FLOAT32_C(  -663.51) },
       INT32_C(         243),
      { EASYSIMD_FLOAT32_C(  -161.19), EASYSIMD_FLOAT32_C(  -536.65), EASYSIMD_FLOAT32_C(   959.15), EASYSIMD_FLOAT32_C(  -663.51) } },
    { { EASYSIMD_FLOAT32_C(   492.72), EASYSIMD_FLOAT32_C(  -162.72), EASYSIMD_FLOAT32_C(  -375.10), EASYSIMD_FLOAT32_C(  -947.45) },
       INT32_C(           4),
      { EASYSIMD_FLOAT32_C(   493.00), EASYSIMD_FLOAT32_C(  -163.00), EASYSIMD_FLOAT32_C(  -375.00), EASYSIMD_FLOAT32_C(  -947.00) } },
    { { EASYSIMD_FLOAT32_C(   728.52), EASYSIMD_FLOAT32_C(   -26.32), EASYSIMD_FLOAT32_C(   638.87), EASYSIMD_FLOAT32_C(   588.91) },
       INT32_C(          20),
      { EASYSIMD_FLOAT32_C(   728.50), EASYSIMD_FLOAT32_C(   -26.50), EASYSIMD_FLOAT32_C(   639.00), EASYSIMD_FLOAT32_C(   589.00) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -914.42), EASYSIMD_FLOAT32_C(   210.42), EASYSIMD_FLOAT32_C(   274.23) },
       INT32_C(          36),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -914.50), EASYSIMD_FLOAT32_C(   210.50), EASYSIMD_FLOAT32_C(   274.25) } },
    { { EASYSIMD_FLOAT32_C(  -560.87),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   127.05), EASYSIMD_FLOAT32_C(  -856.85) },
       INT32_C(          52),
      { EASYSIMD_FLOAT32_C(  -560.88),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   127.00), EASYSIMD_FLOAT32_C(  -856.88) } },
    { { EASYSIMD_FLOAT32_C(   845.91), EASYSIMD_FLOAT32_C(   444.14), EASYSIMD_FLOAT32_C(   807.52), EASYSIMD_FLOAT32_C(  -315.29) },
       INT32_C(          68),
      { EASYSIMD_FLOAT32_C(   845.94), EASYSIMD_FLOAT32_C(   444.12), EASYSIMD_FLOAT32_C(   807.50), EASYSIMD_FLOAT32_C(  -315.31) } },
    { { EASYSIMD_FLOAT32_C(   766.67), EASYSIMD_FLOAT32_C(    21.20), EASYSIMD_FLOAT32_C(   871.67), EASYSIMD_FLOAT32_C(   259.40) },
       INT32_C(          84),
      { EASYSIMD_FLOAT32_C(   766.66), EASYSIMD_FLOAT32_C(    21.19), EASYSIMD_FLOAT32_C(   871.66), EASYSIMD_FLOAT32_C(   259.41) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   311.95), EASYSIMD_FLOAT32_C(  -276.85), EASYSIMD_FLOAT32_C(  -774.90) },
       INT32_C(         100),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   311.95), EASYSIMD_FLOAT32_C(  -276.84), EASYSIMD_FLOAT32_C(  -774.91) } },
    { { EASYSIMD_FLOAT32_C(   814.00), EASYSIMD_FLOAT32_C(   871.38), EASYSIMD_FLOAT32_C(   -55.18), EASYSIMD_FLOAT32_C(   899.58) },
       INT32_C(         116),
      { EASYSIMD_FLOAT32_C(   814.00), EASYSIMD_FLOAT32_C(   871.38), EASYSIMD_FLOAT32_C(   -55.18), EASYSIMD_FLOAT32_C(   899.58) } },
    { { EASYSIMD_FLOAT32_C(  -780.95),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -905.21), EASYSIMD_FLOAT32_C(  -341.82) },
       INT32_C(         132),
      { EASYSIMD_FLOAT32_C(  -780.95),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -905.21), EASYSIMD_FLOAT32_C(  -341.82) } },
    { { EASYSIMD_FLOAT32_C(   983.68), EASYSIMD_FLOAT32_C(  -306.95), EASYSIMD_FLOAT32_C(     9.74), EASYSIMD_FLOAT32_C(   829.59) },
       INT32_C(         148),
      { EASYSIMD_FLOAT32_C(   983.68), EASYSIMD_FLOAT32_C(  -306.95), EASYSIMD_FLOAT32_C(     9.74), EASYSIMD_FLOAT32_C(   829.59) } },
    { { EASYSIMD_FLOAT32_C(  -182.74),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -955.30), EASYSIMD_FLOAT32_C(  -416.07) },
       INT32_C(         164),
      { EASYSIMD_FLOAT32_C(  -182.74),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -955.30), EASYSIMD_FLOAT32_C(  -416.07) } },
    { { EASYSIMD_FLOAT32_C(   393.98),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   155.28), EASYSIMD_FLOAT32_C(  -882.87) },
       INT32_C(         180),
      { EASYSIMD_FLOAT32_C(   393.98),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   155.28), EASYSIMD_FLOAT32_C(  -882.87) } },
    { { EASYSIMD_FLOAT32_C(  -547.96), EASYSIMD_FLOAT32_C(   312.29), EASYSIMD_FLOAT32_C(   423.96), EASYSIMD_FLOAT32_C(  -648.38) },
       INT32_C(         196),
      { EASYSIMD_FLOAT32_C(  -547.96), EASYSIMD_FLOAT32_C(   312.29), EASYSIMD_FLOAT32_C(   423.96), EASYSIMD_FLOAT32_C(  -648.38) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -589.74), EASYSIMD_FLOAT32_C(  -511.12), EASYSIMD_FLOAT32_C(  -698.81) },
       INT32_C(         212),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -589.74), EASYSIMD_FLOAT32_C(  -511.12), EASYSIMD_FLOAT32_C(  -698.81) } },
    { { EASYSIMD_FLOAT32_C(   102.53), EASYSIMD_FLOAT32_C(   372.75), EASYSIMD_FLOAT32_C(  -596.22), EASYSIMD_FLOAT32_C(  -887.73) },
       INT32_C(         228),
      { EASYSIMD_FLOAT32_C(   102.53), EASYSIMD_FLOAT32_C(   372.75), EASYSIMD_FLOAT32_C(  -596.22), EASYSIMD_FLOAT32_C(  -887.73) } },
    { { EASYSIMD_FLOAT32_C(  -459.02), EASYSIMD_FLOAT32_C(   -70.47), EASYSIMD_FLOAT32_C(   716.63), EASYSIMD_FLOAT32_C(  -414.33) },
       INT32_C(         244),
      { EASYSIMD_FLOAT32_C(  -459.02), EASYSIMD_FLOAT32_C(   -70.47), EASYSIMD_FLOAT32_C(   716.63), EASYSIMD_FLOAT32_C(  -414.33) } },
  };

  easysimd__m128 a, r;

  a = easysimd_mm_loadu_ps(test_vec[0].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(           0));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[1].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          16));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[2].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          32));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[3].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          48));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[4].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          64));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[5].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          80));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[6].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          96));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[7].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         112));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[7].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[8].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         128));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[8].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[9].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         144));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[9].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[10].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         160));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[10].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[11].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         176));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[11].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[12].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         192));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[12].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[13].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         208));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[13].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[14].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         224));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[14].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[15].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         240));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[15].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[16].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(           1));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[16].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[17].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          17));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[17].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[18].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          33));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[18].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[19].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          49));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[19].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[20].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          65));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[20].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[21].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          81));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[21].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[22].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          97));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[22].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[23].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         113));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[23].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[24].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         129));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[24].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[25].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         145));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[25].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[26].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         161));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[26].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[27].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         177));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[27].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[28].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         193));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[28].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[29].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         209));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[29].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[30].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         225));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[30].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[31].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         241));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[31].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[32].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(           2));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[32].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[33].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          18));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[33].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[34].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          34));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[34].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[35].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          50));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[35].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[36].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          66));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[36].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[37].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          82));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[37].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[38].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          98));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[38].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[39].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         114));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[39].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[40].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         130));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[40].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[41].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         146));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[41].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[42].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         162));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[42].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[43].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         178));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[43].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[44].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         194));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[44].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[45].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         210));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[45].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[46].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         226));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[46].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[47].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         242));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[47].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[48].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(           3));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[48].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[49].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          19));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[49].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[50].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          35));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[50].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[51].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          51));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[51].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[52].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          67));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[52].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[53].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          83));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[53].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[54].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          99));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[54].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[55].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         115));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[55].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[56].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         131));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[56].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[57].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         147));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[57].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[58].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         163));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[58].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[59].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         179));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[59].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[60].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         195));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[60].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[61].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         211));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[61].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[62].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         227));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[62].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[63].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         243));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[63].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[64].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(           4));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[64].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[65].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          20));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[65].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[66].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          36));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[66].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[67].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          52));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[67].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[68].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          68));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[68].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[69].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(          84));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[69].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[70].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         100));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[70].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[71].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         116));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[71].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[72].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         132));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[72].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[73].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         148));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[73].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[74].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         164));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[74].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[75].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         180));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[75].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[76].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         196));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[76].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[77].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         212));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[77].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[78].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         228));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[78].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[79].a);
  r = easysimd_mm_roundscale_ps(a, INT32_C(         244));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[79].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 5 ; i++) {
    for (int j = 0 ; j < 16 ; j++) {
      easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
      if((easysimd_test_codegen_rand() & 1)) {
        if((easysimd_test_codegen_rand() & 1))
          a = easysimd_mm_blend_ps(a, easysimd_mm_set1_ps(EASYSIMD_MATH_NANF), 1);
        else {
          if((easysimd_test_codegen_rand() & 1))
            a = easysimd_mm_blend_ps(a, easysimd_mm_set1_ps(EASYSIMD_MATH_INFINITY), 2);
          else
            a = easysimd_mm_blend_ps(a, easysimd_mm_set1_ps(-EASYSIMD_MATH_INFINITY), 2);
        }
      }
      int imm8 = ((j << 4) | round_type[i]) & 255;
      easysimd__m128 r;
      EASYSIMD_CONSTIFY_256_(easysimd_mm_roundscale_ps, r, easysimd_mm_setzero_ps(), imm8, a);

      easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
      easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
      easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
    }
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_roundscale_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[4];
    const easysimd__mmask8 k;
    const easysimd_float32 a[4];
    const int32_t imm8;
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -316.31), EASYSIMD_FLOAT32_C(   649.73), EASYSIMD_FLOAT32_C(   200.80), EASYSIMD_FLOAT32_C(   -79.92) },
      UINT8_C(161),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   963.57), EASYSIMD_FLOAT32_C(  -663.71), EASYSIMD_FLOAT32_C(  -906.90) },
       INT32_C(         112),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   649.73), EASYSIMD_FLOAT32_C(   200.80), EASYSIMD_FLOAT32_C(   -79.92) } },
    { { EASYSIMD_FLOAT32_C(  -584.01), EASYSIMD_FLOAT32_C(  -683.67), EASYSIMD_FLOAT32_C(   359.10), EASYSIMD_FLOAT32_C(   539.12) },
      UINT8_C( 84),
      { EASYSIMD_FLOAT32_C(  -642.45), EASYSIMD_FLOAT32_C(   980.52), EASYSIMD_FLOAT32_C(   709.42), EASYSIMD_FLOAT32_C(  -815.38) },
       INT32_C(          49),
      { EASYSIMD_FLOAT32_C(  -584.01), EASYSIMD_FLOAT32_C(  -683.67), EASYSIMD_FLOAT32_C(   709.38), EASYSIMD_FLOAT32_C(   539.12) } },
    { { EASYSIMD_FLOAT32_C(  -716.00), EASYSIMD_FLOAT32_C(  -111.36), EASYSIMD_FLOAT32_C(  -489.38), EASYSIMD_FLOAT32_C(   896.66) },
      UINT8_C(244),
      { EASYSIMD_FLOAT32_C(   -57.08), EASYSIMD_FLOAT32_C(  -626.52), EASYSIMD_FLOAT32_C(  -249.57), EASYSIMD_FLOAT32_C(   626.60) },
       INT32_C(          18),
      { EASYSIMD_FLOAT32_C(  -716.00), EASYSIMD_FLOAT32_C(  -111.36), EASYSIMD_FLOAT32_C(  -249.50), EASYSIMD_FLOAT32_C(   896.66) } },
    { { EASYSIMD_FLOAT32_C(  -453.32), EASYSIMD_FLOAT32_C(   134.70), EASYSIMD_FLOAT32_C(   315.58), EASYSIMD_FLOAT32_C(  -489.75) },
      UINT8_C(109),
      { EASYSIMD_FLOAT32_C(   408.68), EASYSIMD_FLOAT32_C(   792.03), EASYSIMD_FLOAT32_C(   887.75), EASYSIMD_FLOAT32_C(   573.55) },
       INT32_C(           3),
      { EASYSIMD_FLOAT32_C(   408.00), EASYSIMD_FLOAT32_C(   134.70), EASYSIMD_FLOAT32_C(   887.00), EASYSIMD_FLOAT32_C(   573.00) } },
    { { EASYSIMD_FLOAT32_C(   -67.35), EASYSIMD_FLOAT32_C(   747.13), EASYSIMD_FLOAT32_C(  -632.66), EASYSIMD_FLOAT32_C(   290.20) },
      UINT8_C(224),
      { EASYSIMD_FLOAT32_C(  -923.24),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   963.50), EASYSIMD_FLOAT32_C(  -799.60) },
       INT32_C(         148),
      { EASYSIMD_FLOAT32_C(   -67.35), EASYSIMD_FLOAT32_C(   747.13), EASYSIMD_FLOAT32_C(  -632.66), EASYSIMD_FLOAT32_C(   290.20) } },
    { { EASYSIMD_FLOAT32_C(  -984.94), EASYSIMD_FLOAT32_C(   653.94), EASYSIMD_FLOAT32_C(  -971.04), EASYSIMD_FLOAT32_C(  -234.51) },
      UINT8_C(123),
      { EASYSIMD_FLOAT32_C(  -947.82),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   827.23), EASYSIMD_FLOAT32_C(   186.88) },
       INT32_C(          64),
      { EASYSIMD_FLOAT32_C(  -947.81),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -971.04), EASYSIMD_FLOAT32_C(   186.88) } },
    { { EASYSIMD_FLOAT32_C(  -870.49), EASYSIMD_FLOAT32_C(  -454.38), EASYSIMD_FLOAT32_C(    14.54), EASYSIMD_FLOAT32_C(  -662.48) },
      UINT8_C( 52),
      { EASYSIMD_FLOAT32_C(   947.19),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   117.04), EASYSIMD_FLOAT32_C(   237.39) },
       INT32_C(          65),
      { EASYSIMD_FLOAT32_C(  -870.49), EASYSIMD_FLOAT32_C(  -454.38), EASYSIMD_FLOAT32_C(   117.00), EASYSIMD_FLOAT32_C(  -662.48) } },
    { { EASYSIMD_FLOAT32_C(   394.20), EASYSIMD_FLOAT32_C(  -528.97), EASYSIMD_FLOAT32_C(  -372.06), EASYSIMD_FLOAT32_C(  -894.77) },
      UINT8_C( 34),
      { EASYSIMD_FLOAT32_C(  -357.00),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -844.53), EASYSIMD_FLOAT32_C(   408.49) },
       INT32_C(          34),
      { EASYSIMD_FLOAT32_C(   394.20),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -372.06), EASYSIMD_FLOAT32_C(  -894.77) } },
  };

  easysimd__m128 src, a, r;

  src = easysimd_mm_loadu_ps(test_vec[0].src);
  a = easysimd_mm_loadu_ps(test_vec[0].a);
  r = easysimd_mm_mask_roundscale_ps(src, test_vec[0].k, a, INT32_C(         112));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[0].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[1].src);
  a = easysimd_mm_loadu_ps(test_vec[1].a);
  r = easysimd_mm_mask_roundscale_ps(src, test_vec[1].k, a, INT32_C(          49));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[1].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[2].src);
  a = easysimd_mm_loadu_ps(test_vec[2].a);
  r = easysimd_mm_mask_roundscale_ps(src, test_vec[2].k, a, INT32_C(          18));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[2].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[3].src);
  a = easysimd_mm_loadu_ps(test_vec[3].a);
  r = easysimd_mm_mask_roundscale_ps(src, test_vec[3].k, a, INT32_C(           3));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[3].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[4].src);
  a = easysimd_mm_loadu_ps(test_vec[4].a);
  r = easysimd_mm_mask_roundscale_ps(src, test_vec[4].k, a, INT32_C(         148));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[4].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[5].src);
  a = easysimd_mm_loadu_ps(test_vec[5].a);
  r = easysimd_mm_mask_roundscale_ps(src, test_vec[5].k, a, INT32_C(          64));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[5].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[6].src);
  a = easysimd_mm_loadu_ps(test_vec[6].a);
  r = easysimd_mm_mask_roundscale_ps(src, test_vec[6].k, a, INT32_C(          65));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[6].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[7].src);
  a = easysimd_mm_loadu_ps(test_vec[7].a);
  r = easysimd_mm_mask_roundscale_ps(src, test_vec[7].k, a, INT32_C(          34));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    if((easysimd_test_codegen_rand() & 1)) {
      if((easysimd_test_codegen_rand() & 1))
        a = easysimd_mm_blend_ps(a, easysimd_mm_set1_ps(EASYSIMD_MATH_NANF), 1);
      else {
        if((easysimd_test_codegen_rand() & 1))
          a = easysimd_mm_blend_ps(a, easysimd_mm_set1_ps(EASYSIMD_MATH_INFINITY), 2);
        else
          a = easysimd_mm_blend_ps(a, easysimd_mm_set1_ps(-EASYSIMD_MATH_INFINITY), 2);
      }
    }
    int imm8 = (((easysimd_test_codegen_rand() & 15) << 4) | round_type[i % 5]) & 255;
    easysimd__m128 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm_mask_roundscale_ps, r, easysimd_mm_setzero_ps(), imm8, src, k, a);

    easysimd_test_x86_write_f32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_roundscale_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float32 a[4];
    const int32_t imm8;
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C(155),
      { EASYSIMD_FLOAT32_C(  -842.49), EASYSIMD_FLOAT32_C(   204.44), EASYSIMD_FLOAT32_C(  -947.60), EASYSIMD_FLOAT32_C(   598.50) },
       INT32_C(         176),
      { EASYSIMD_FLOAT32_C(  -842.49), EASYSIMD_FLOAT32_C(   204.44), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   598.50) } },
    { UINT8_C(208),
      { EASYSIMD_FLOAT32_C(   671.48), EASYSIMD_FLOAT32_C(   347.71), EASYSIMD_FLOAT32_C(  -439.78), EASYSIMD_FLOAT32_C(   756.13) },
       INT32_C(         161),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(139),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -490.18), EASYSIMD_FLOAT32_C(   344.22), EASYSIMD_FLOAT32_C(    52.75) },
       INT32_C(          82),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -490.16), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    52.75) } },
    { UINT8_C( 73),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   -82.84), EASYSIMD_FLOAT32_C(   262.82), EASYSIMD_FLOAT32_C(  -976.36) },
       INT32_C(         195),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -976.36) } },
    { UINT8_C(175),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -693.64), EASYSIMD_FLOAT32_C(  -971.72), EASYSIMD_FLOAT32_C(   -82.61) },
       INT32_C(          52),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -693.62), EASYSIMD_FLOAT32_C(  -971.75), EASYSIMD_FLOAT32_C(   -82.62) } },
    { UINT8_C( 93),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -136.88), EASYSIMD_FLOAT32_C(    78.11), EASYSIMD_FLOAT32_C(  -210.16) },
       INT32_C(          16),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    78.00), EASYSIMD_FLOAT32_C(  -210.00) } },
    { UINT8_C(240),
      { EASYSIMD_FLOAT32_C(   385.54), EASYSIMD_FLOAT32_C(   702.48), EASYSIMD_FLOAT32_C(  -960.83), EASYSIMD_FLOAT32_C(  -633.62) },
       INT32_C(         225),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(112),
      { EASYSIMD_FLOAT32_C(  -710.21),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -263.45), EASYSIMD_FLOAT32_C(  -686.56) },
       INT32_C(         226),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
  };

  easysimd__m128 a, r;

  a = easysimd_mm_loadu_ps(test_vec[0].a);
  r = easysimd_mm_maskz_roundscale_ps(test_vec[0].k, a, INT32_C(         176));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[1].a);
  r = easysimd_mm_maskz_roundscale_ps(test_vec[1].k, a, INT32_C(         161));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[2].a);
  r = easysimd_mm_maskz_roundscale_ps(test_vec[2].k, a, INT32_C(          82));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[3].a);
  r = easysimd_mm_maskz_roundscale_ps(test_vec[3].k, a, INT32_C(         195));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[4].a);
  r = easysimd_mm_maskz_roundscale_ps(test_vec[4].k, a, INT32_C(          52));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[5].a);
  r = easysimd_mm_maskz_roundscale_ps(test_vec[5].k, a, INT32_C(          16));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[6].a);
  r = easysimd_mm_maskz_roundscale_ps(test_vec[6].k, a, INT32_C(         225));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[7].a);
  r = easysimd_mm_maskz_roundscale_ps(test_vec[7].k, a, INT32_C(         226));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    if((easysimd_test_codegen_rand() & 1)) {
      if((easysimd_test_codegen_rand() & 1))
        a = easysimd_mm_blend_ps(a, easysimd_mm_set1_ps(EASYSIMD_MATH_NANF), 1);
      else {
        if((easysimd_test_codegen_rand() & 1))
          a = easysimd_mm_blend_ps(a, easysimd_mm_set1_ps(EASYSIMD_MATH_INFINITY), 2);
        else
          a = easysimd_mm_blend_ps(a, easysimd_mm_set1_ps(-EASYSIMD_MATH_INFINITY), 2);
      }
    }
    int imm8 = ((easysimd_test_codegen_rand() & 15) << 4 | round_type[i % 5]) & 255;
    easysimd__m128 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm_maskz_roundscale_ps, r, easysimd_mm_setzero_ps(), imm8, k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_roundscale_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[8];
    const int32_t imm8;
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   680.63),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   441.98), EASYSIMD_FLOAT32_C(   661.29),
        EASYSIMD_FLOAT32_C(    38.40), EASYSIMD_FLOAT32_C(  -974.53), EASYSIMD_FLOAT32_C(   579.66), EASYSIMD_FLOAT32_C(  -989.32) },
       INT32_C(           0),
      { EASYSIMD_FLOAT32_C(   681.00),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   442.00), EASYSIMD_FLOAT32_C(   661.00),
        EASYSIMD_FLOAT32_C(    38.00), EASYSIMD_FLOAT32_C(  -975.00), EASYSIMD_FLOAT32_C(   580.00), EASYSIMD_FLOAT32_C(  -989.00) } },
    { { EASYSIMD_FLOAT32_C(  -684.35), EASYSIMD_FLOAT32_C(   -10.18), EASYSIMD_FLOAT32_C(  -771.27), EASYSIMD_FLOAT32_C(  -448.98),
        EASYSIMD_FLOAT32_C(   301.51), EASYSIMD_FLOAT32_C(   239.71), EASYSIMD_FLOAT32_C(   730.34), EASYSIMD_FLOAT32_C(   112.35) },
       INT32_C(          16),
      { EASYSIMD_FLOAT32_C(  -684.50), EASYSIMD_FLOAT32_C(   -10.00), EASYSIMD_FLOAT32_C(  -771.50), EASYSIMD_FLOAT32_C(  -449.00),
        EASYSIMD_FLOAT32_C(   301.50), EASYSIMD_FLOAT32_C(   239.50), EASYSIMD_FLOAT32_C(   730.50), EASYSIMD_FLOAT32_C(   112.50) } },
    { { EASYSIMD_FLOAT32_C(   927.48), EASYSIMD_FLOAT32_C(  -663.37), EASYSIMD_FLOAT32_C(  -126.17), EASYSIMD_FLOAT32_C(   917.27),
        EASYSIMD_FLOAT32_C(   824.25), EASYSIMD_FLOAT32_C(  -774.04), EASYSIMD_FLOAT32_C(   704.40), EASYSIMD_FLOAT32_C(  -459.07) },
       INT32_C(          32),
      { EASYSIMD_FLOAT32_C(   927.50), EASYSIMD_FLOAT32_C(  -663.25), EASYSIMD_FLOAT32_C(  -126.25), EASYSIMD_FLOAT32_C(   917.25),
        EASYSIMD_FLOAT32_C(   824.25), EASYSIMD_FLOAT32_C(  -774.00), EASYSIMD_FLOAT32_C(   704.50), EASYSIMD_FLOAT32_C(  -459.00) } },
    { { EASYSIMD_FLOAT32_C(  -841.48),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   -22.59), EASYSIMD_FLOAT32_C(   528.01),
        EASYSIMD_FLOAT32_C(    -3.94), EASYSIMD_FLOAT32_C(  -361.30), EASYSIMD_FLOAT32_C(  -433.59), EASYSIMD_FLOAT32_C(    21.54) },
       INT32_C(          48),
      { EASYSIMD_FLOAT32_C(  -841.50),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   -22.62), EASYSIMD_FLOAT32_C(   528.00),
        EASYSIMD_FLOAT32_C(    -4.00), EASYSIMD_FLOAT32_C(  -361.25), EASYSIMD_FLOAT32_C(  -433.62), EASYSIMD_FLOAT32_C(    21.50) } },
    { { EASYSIMD_FLOAT32_C(  -339.97), EASYSIMD_FLOAT32_C(   658.60), EASYSIMD_FLOAT32_C(   408.96), EASYSIMD_FLOAT32_C(   649.85),
        EASYSIMD_FLOAT32_C(   887.33), EASYSIMD_FLOAT32_C(   959.98), EASYSIMD_FLOAT32_C(   -48.64), EASYSIMD_FLOAT32_C(   127.05) },
       INT32_C(          64),
      { EASYSIMD_FLOAT32_C(  -340.00), EASYSIMD_FLOAT32_C(   658.62), EASYSIMD_FLOAT32_C(   408.94), EASYSIMD_FLOAT32_C(   649.88),
        EASYSIMD_FLOAT32_C(   887.31), EASYSIMD_FLOAT32_C(   960.00), EASYSIMD_FLOAT32_C(   -48.62), EASYSIMD_FLOAT32_C(   127.06) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -528.44), EASYSIMD_FLOAT32_C(   617.81), EASYSIMD_FLOAT32_C(  -599.66),
        EASYSIMD_FLOAT32_C(   345.39), EASYSIMD_FLOAT32_C(   535.08), EASYSIMD_FLOAT32_C(  -775.41), EASYSIMD_FLOAT32_C(   571.35) },
       INT32_C(          80),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -528.44), EASYSIMD_FLOAT32_C(   617.81), EASYSIMD_FLOAT32_C(  -599.66),
        EASYSIMD_FLOAT32_C(   345.38), EASYSIMD_FLOAT32_C(   535.09), EASYSIMD_FLOAT32_C(  -775.41), EASYSIMD_FLOAT32_C(   571.34) } },
    { { EASYSIMD_FLOAT32_C(  -131.87), EASYSIMD_FLOAT32_C(   397.99), EASYSIMD_FLOAT32_C(  -680.39), EASYSIMD_FLOAT32_C(   845.53),
        EASYSIMD_FLOAT32_C(   -74.00), EASYSIMD_FLOAT32_C(   315.67), EASYSIMD_FLOAT32_C(  -515.77), EASYSIMD_FLOAT32_C(   492.41) },
       INT32_C(          96),
      { EASYSIMD_FLOAT32_C(  -131.88), EASYSIMD_FLOAT32_C(   397.98), EASYSIMD_FLOAT32_C(  -680.39), EASYSIMD_FLOAT32_C(   845.53),
        EASYSIMD_FLOAT32_C(   -74.00), EASYSIMD_FLOAT32_C(   315.67), EASYSIMD_FLOAT32_C(  -515.77), EASYSIMD_FLOAT32_C(   492.41) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -930.49), EASYSIMD_FLOAT32_C(   430.51), EASYSIMD_FLOAT32_C(   362.62),
        EASYSIMD_FLOAT32_C(   728.11), EASYSIMD_FLOAT32_C(  -160.53), EASYSIMD_FLOAT32_C(    12.46), EASYSIMD_FLOAT32_C(   615.44) },
       INT32_C(         112),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -930.49), EASYSIMD_FLOAT32_C(   430.51), EASYSIMD_FLOAT32_C(   362.62),
        EASYSIMD_FLOAT32_C(   728.11), EASYSIMD_FLOAT32_C(  -160.53), EASYSIMD_FLOAT32_C(    12.46), EASYSIMD_FLOAT32_C(   615.44) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -510.23), EASYSIMD_FLOAT32_C(  -972.46), EASYSIMD_FLOAT32_C(   214.05),
        EASYSIMD_FLOAT32_C(  -892.43), EASYSIMD_FLOAT32_C(  -572.12), EASYSIMD_FLOAT32_C(  -440.56), EASYSIMD_FLOAT32_C(   642.65) },
       INT32_C(         128),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -510.23), EASYSIMD_FLOAT32_C(  -972.46), EASYSIMD_FLOAT32_C(   214.05),
        EASYSIMD_FLOAT32_C(  -892.43), EASYSIMD_FLOAT32_C(  -572.12), EASYSIMD_FLOAT32_C(  -440.56), EASYSIMD_FLOAT32_C(   642.65) } },
    { { EASYSIMD_FLOAT32_C(  -117.87), EASYSIMD_FLOAT32_C(   418.00), EASYSIMD_FLOAT32_C(    -1.08), EASYSIMD_FLOAT32_C(  -719.88),
        EASYSIMD_FLOAT32_C(   737.60), EASYSIMD_FLOAT32_C(  -155.55), EASYSIMD_FLOAT32_C(   206.13), EASYSIMD_FLOAT32_C(    53.27) },
       INT32_C(         144),
      { EASYSIMD_FLOAT32_C(  -117.87), EASYSIMD_FLOAT32_C(   418.00), EASYSIMD_FLOAT32_C(    -1.08), EASYSIMD_FLOAT32_C(  -719.88),
        EASYSIMD_FLOAT32_C(   737.60), EASYSIMD_FLOAT32_C(  -155.55), EASYSIMD_FLOAT32_C(   206.13), EASYSIMD_FLOAT32_C(    53.27) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   390.47), EASYSIMD_FLOAT32_C(  -968.73), EASYSIMD_FLOAT32_C(  -231.95),
        EASYSIMD_FLOAT32_C(  -179.02), EASYSIMD_FLOAT32_C(   393.89), EASYSIMD_FLOAT32_C(  -503.84), EASYSIMD_FLOAT32_C(   660.45) },
       INT32_C(         160),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   390.47), EASYSIMD_FLOAT32_C(  -968.73), EASYSIMD_FLOAT32_C(  -231.95),
        EASYSIMD_FLOAT32_C(  -179.02), EASYSIMD_FLOAT32_C(   393.89), EASYSIMD_FLOAT32_C(  -503.84), EASYSIMD_FLOAT32_C(   660.45) } },
    { { EASYSIMD_FLOAT32_C(  -540.10), EASYSIMD_FLOAT32_C(  -629.83), EASYSIMD_FLOAT32_C(  -145.91), EASYSIMD_FLOAT32_C(   -50.34),
        EASYSIMD_FLOAT32_C(  -602.29), EASYSIMD_FLOAT32_C(  -931.86), EASYSIMD_FLOAT32_C(    57.24), EASYSIMD_FLOAT32_C(  -174.41) },
       INT32_C(         176),
      { EASYSIMD_FLOAT32_C(  -540.10), EASYSIMD_FLOAT32_C(  -629.83), EASYSIMD_FLOAT32_C(  -145.91), EASYSIMD_FLOAT32_C(   -50.34),
        EASYSIMD_FLOAT32_C(  -602.29), EASYSIMD_FLOAT32_C(  -931.86), EASYSIMD_FLOAT32_C(    57.24), EASYSIMD_FLOAT32_C(  -174.41) } },
    { { EASYSIMD_FLOAT32_C(  -300.11), EASYSIMD_FLOAT32_C(   478.06), EASYSIMD_FLOAT32_C(  -241.63), EASYSIMD_FLOAT32_C(   582.02),
        EASYSIMD_FLOAT32_C(  -103.95), EASYSIMD_FLOAT32_C(   757.29), EASYSIMD_FLOAT32_C(   862.15), EASYSIMD_FLOAT32_C(  -366.35) },
       INT32_C(         192),
      { EASYSIMD_FLOAT32_C(  -300.11), EASYSIMD_FLOAT32_C(   478.06), EASYSIMD_FLOAT32_C(  -241.63), EASYSIMD_FLOAT32_C(   582.02),
        EASYSIMD_FLOAT32_C(  -103.95), EASYSIMD_FLOAT32_C(   757.29), EASYSIMD_FLOAT32_C(   862.15), EASYSIMD_FLOAT32_C(  -366.35) } },
    { { EASYSIMD_FLOAT32_C(    68.27), EASYSIMD_FLOAT32_C(   686.92), EASYSIMD_FLOAT32_C(   930.42), EASYSIMD_FLOAT32_C(   766.81),
        EASYSIMD_FLOAT32_C(    77.39), EASYSIMD_FLOAT32_C(   961.69), EASYSIMD_FLOAT32_C(  -465.14), EASYSIMD_FLOAT32_C(   898.37) },
       INT32_C(         208),
      { EASYSIMD_FLOAT32_C(    68.27), EASYSIMD_FLOAT32_C(   686.92), EASYSIMD_FLOAT32_C(   930.42), EASYSIMD_FLOAT32_C(   766.81),
        EASYSIMD_FLOAT32_C(    77.39), EASYSIMD_FLOAT32_C(   961.69), EASYSIMD_FLOAT32_C(  -465.14), EASYSIMD_FLOAT32_C(   898.37) } },
    { { EASYSIMD_FLOAT32_C(    31.02), EASYSIMD_FLOAT32_C(   558.82), EASYSIMD_FLOAT32_C(   761.92), EASYSIMD_FLOAT32_C(   142.63),
        EASYSIMD_FLOAT32_C(  -981.28), EASYSIMD_FLOAT32_C(  -867.91), EASYSIMD_FLOAT32_C(   996.72), EASYSIMD_FLOAT32_C(   -31.62) },
       INT32_C(         224),
      { EASYSIMD_FLOAT32_C(    31.02), EASYSIMD_FLOAT32_C(   558.82), EASYSIMD_FLOAT32_C(   761.92), EASYSIMD_FLOAT32_C(   142.63),
        EASYSIMD_FLOAT32_C(  -981.28), EASYSIMD_FLOAT32_C(  -867.91), EASYSIMD_FLOAT32_C(   996.72), EASYSIMD_FLOAT32_C(   -31.62) } },
    { { EASYSIMD_FLOAT32_C(  -935.13),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   355.39), EASYSIMD_FLOAT32_C(  -307.55),
        EASYSIMD_FLOAT32_C(  -274.48), EASYSIMD_FLOAT32_C(  -166.56), EASYSIMD_FLOAT32_C(   450.82), EASYSIMD_FLOAT32_C(  -692.46) },
       INT32_C(         240),
      { EASYSIMD_FLOAT32_C(  -935.13),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   355.39), EASYSIMD_FLOAT32_C(  -307.55),
        EASYSIMD_FLOAT32_C(  -274.48), EASYSIMD_FLOAT32_C(  -166.56), EASYSIMD_FLOAT32_C(   450.82), EASYSIMD_FLOAT32_C(  -692.46) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   809.84), EASYSIMD_FLOAT32_C(   237.96), EASYSIMD_FLOAT32_C(  -949.93),
        EASYSIMD_FLOAT32_C(   740.26), EASYSIMD_FLOAT32_C(     4.77), EASYSIMD_FLOAT32_C(   127.46), EASYSIMD_FLOAT32_C(   701.95) },
       INT32_C(           1),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   809.00), EASYSIMD_FLOAT32_C(   237.00), EASYSIMD_FLOAT32_C(  -950.00),
        EASYSIMD_FLOAT32_C(   740.00), EASYSIMD_FLOAT32_C(     4.00), EASYSIMD_FLOAT32_C(   127.00), EASYSIMD_FLOAT32_C(   701.00) } },
    { { EASYSIMD_FLOAT32_C(    57.52), EASYSIMD_FLOAT32_C(  -429.34), EASYSIMD_FLOAT32_C(  -415.34), EASYSIMD_FLOAT32_C(  -180.55),
        EASYSIMD_FLOAT32_C(   713.28), EASYSIMD_FLOAT32_C(  -396.62), EASYSIMD_FLOAT32_C(   -48.46), EASYSIMD_FLOAT32_C(   710.01) },
       INT32_C(          17),
      { EASYSIMD_FLOAT32_C(    57.50), EASYSIMD_FLOAT32_C(  -429.50), EASYSIMD_FLOAT32_C(  -415.50), EASYSIMD_FLOAT32_C(  -181.00),
        EASYSIMD_FLOAT32_C(   713.00), EASYSIMD_FLOAT32_C(  -397.00), EASYSIMD_FLOAT32_C(   -48.50), EASYSIMD_FLOAT32_C(   710.00) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   774.87), EASYSIMD_FLOAT32_C(   597.38), EASYSIMD_FLOAT32_C(  -163.27),
        EASYSIMD_FLOAT32_C(  -532.68), EASYSIMD_FLOAT32_C(  -677.10), EASYSIMD_FLOAT32_C(   670.17), EASYSIMD_FLOAT32_C(   918.14) },
       INT32_C(          33),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   774.75), EASYSIMD_FLOAT32_C(   597.25), EASYSIMD_FLOAT32_C(  -163.50),
        EASYSIMD_FLOAT32_C(  -532.75), EASYSIMD_FLOAT32_C(  -677.25), EASYSIMD_FLOAT32_C(   670.00), EASYSIMD_FLOAT32_C(   918.00) } },
    { { EASYSIMD_FLOAT32_C(   126.25), EASYSIMD_FLOAT32_C(  -199.87), EASYSIMD_FLOAT32_C(   762.81), EASYSIMD_FLOAT32_C(   -63.91),
        EASYSIMD_FLOAT32_C(  -961.91), EASYSIMD_FLOAT32_C(   812.88), EASYSIMD_FLOAT32_C(  -323.65), EASYSIMD_FLOAT32_C(    42.86) },
       INT32_C(          49),
      { EASYSIMD_FLOAT32_C(   126.25), EASYSIMD_FLOAT32_C(  -199.88), EASYSIMD_FLOAT32_C(   762.75), EASYSIMD_FLOAT32_C(   -64.00),
        EASYSIMD_FLOAT32_C(  -962.00), EASYSIMD_FLOAT32_C(   812.88), EASYSIMD_FLOAT32_C(  -323.75), EASYSIMD_FLOAT32_C(    42.75) } },
    { { EASYSIMD_FLOAT32_C(  -621.70), EASYSIMD_FLOAT32_C(  -417.51), EASYSIMD_FLOAT32_C(   966.17), EASYSIMD_FLOAT32_C(   435.83),
        EASYSIMD_FLOAT32_C(   153.15), EASYSIMD_FLOAT32_C(  -449.17), EASYSIMD_FLOAT32_C(  -744.73), EASYSIMD_FLOAT32_C(  -133.57) },
       INT32_C(          65),
      { EASYSIMD_FLOAT32_C(  -621.75), EASYSIMD_FLOAT32_C(  -417.56), EASYSIMD_FLOAT32_C(   966.12), EASYSIMD_FLOAT32_C(   435.81),
        EASYSIMD_FLOAT32_C(   153.12), EASYSIMD_FLOAT32_C(  -449.19), EASYSIMD_FLOAT32_C(  -744.75), EASYSIMD_FLOAT32_C(  -133.62) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -423.56), EASYSIMD_FLOAT32_C(  -274.03), EASYSIMD_FLOAT32_C(  -311.84),
        EASYSIMD_FLOAT32_C(  -648.69), EASYSIMD_FLOAT32_C(  -676.65), EASYSIMD_FLOAT32_C(   524.89), EASYSIMD_FLOAT32_C(  -181.37) },
       INT32_C(          81),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -423.56), EASYSIMD_FLOAT32_C(  -274.03), EASYSIMD_FLOAT32_C(  -311.84),
        EASYSIMD_FLOAT32_C(  -648.72), EASYSIMD_FLOAT32_C(  -676.66), EASYSIMD_FLOAT32_C(   524.88), EASYSIMD_FLOAT32_C(  -181.38) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   276.69), EASYSIMD_FLOAT32_C(  -405.28), EASYSIMD_FLOAT32_C(   863.01),
        EASYSIMD_FLOAT32_C(  -923.18), EASYSIMD_FLOAT32_C(  -642.46), EASYSIMD_FLOAT32_C(  -200.90), EASYSIMD_FLOAT32_C(  -885.09) },
       INT32_C(          97),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   276.69), EASYSIMD_FLOAT32_C(  -405.28), EASYSIMD_FLOAT32_C(   863.00),
        EASYSIMD_FLOAT32_C(  -923.19), EASYSIMD_FLOAT32_C(  -642.47), EASYSIMD_FLOAT32_C(  -200.91), EASYSIMD_FLOAT32_C(  -885.09) } },
    { { EASYSIMD_FLOAT32_C(   157.77), EASYSIMD_FLOAT32_C(   110.75), EASYSIMD_FLOAT32_C(   853.76), EASYSIMD_FLOAT32_C(   740.26),
        EASYSIMD_FLOAT32_C(    76.93), EASYSIMD_FLOAT32_C(   289.58), EASYSIMD_FLOAT32_C(  -106.60), EASYSIMD_FLOAT32_C(   627.76) },
       INT32_C(         113),
      { EASYSIMD_FLOAT32_C(   157.77), EASYSIMD_FLOAT32_C(   110.75), EASYSIMD_FLOAT32_C(   853.76), EASYSIMD_FLOAT32_C(   740.26),
        EASYSIMD_FLOAT32_C(    76.93), EASYSIMD_FLOAT32_C(   289.58), EASYSIMD_FLOAT32_C(  -106.60), EASYSIMD_FLOAT32_C(   627.76) } },
    { { EASYSIMD_FLOAT32_C(   759.84), EASYSIMD_FLOAT32_C(  -218.03), EASYSIMD_FLOAT32_C(  -248.33), EASYSIMD_FLOAT32_C(  -663.72),
        EASYSIMD_FLOAT32_C(   507.94), EASYSIMD_FLOAT32_C(   439.83), EASYSIMD_FLOAT32_C(  -312.42), EASYSIMD_FLOAT32_C(   831.29) },
       INT32_C(         129),
      { EASYSIMD_FLOAT32_C(   759.84), EASYSIMD_FLOAT32_C(  -218.03), EASYSIMD_FLOAT32_C(  -248.33), EASYSIMD_FLOAT32_C(  -663.72),
        EASYSIMD_FLOAT32_C(   507.94), EASYSIMD_FLOAT32_C(   439.83), EASYSIMD_FLOAT32_C(  -312.42), EASYSIMD_FLOAT32_C(   831.29) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -522.46), EASYSIMD_FLOAT32_C(  -840.22), EASYSIMD_FLOAT32_C(  -757.02),
        EASYSIMD_FLOAT32_C(   754.24), EASYSIMD_FLOAT32_C(  -245.50), EASYSIMD_FLOAT32_C(  -894.01), EASYSIMD_FLOAT32_C(   831.06) },
       INT32_C(         145),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -522.46), EASYSIMD_FLOAT32_C(  -840.22), EASYSIMD_FLOAT32_C(  -757.02),
        EASYSIMD_FLOAT32_C(   754.24), EASYSIMD_FLOAT32_C(  -245.50), EASYSIMD_FLOAT32_C(  -894.01), EASYSIMD_FLOAT32_C(   831.06) } },
    { { EASYSIMD_FLOAT32_C(   945.97), EASYSIMD_FLOAT32_C(   282.45), EASYSIMD_FLOAT32_C(  -619.45), EASYSIMD_FLOAT32_C(   103.74),
        EASYSIMD_FLOAT32_C(  -606.79), EASYSIMD_FLOAT32_C(  -765.70), EASYSIMD_FLOAT32_C(  -156.01), EASYSIMD_FLOAT32_C(   470.14) },
       INT32_C(         161),
      { EASYSIMD_FLOAT32_C(   945.97), EASYSIMD_FLOAT32_C(   282.45), EASYSIMD_FLOAT32_C(  -619.45), EASYSIMD_FLOAT32_C(   103.74),
        EASYSIMD_FLOAT32_C(  -606.79), EASYSIMD_FLOAT32_C(  -765.70), EASYSIMD_FLOAT32_C(  -156.01), EASYSIMD_FLOAT32_C(   470.14) } },
    { { EASYSIMD_FLOAT32_C(   737.40), EASYSIMD_FLOAT32_C(    97.90), EASYSIMD_FLOAT32_C(    68.75), EASYSIMD_FLOAT32_C(   497.23),
        EASYSIMD_FLOAT32_C(   879.86), EASYSIMD_FLOAT32_C(   820.42), EASYSIMD_FLOAT32_C(   833.51), EASYSIMD_FLOAT32_C(   387.80) },
       INT32_C(         177),
      { EASYSIMD_FLOAT32_C(   737.40), EASYSIMD_FLOAT32_C(    97.90), EASYSIMD_FLOAT32_C(    68.75), EASYSIMD_FLOAT32_C(   497.23),
        EASYSIMD_FLOAT32_C(   879.86), EASYSIMD_FLOAT32_C(   820.42), EASYSIMD_FLOAT32_C(   833.51), EASYSIMD_FLOAT32_C(   387.80) } },
    { { EASYSIMD_FLOAT32_C(  -478.91), EASYSIMD_FLOAT32_C(   219.09), EASYSIMD_FLOAT32_C(  -775.03), EASYSIMD_FLOAT32_C(  -972.69),
        EASYSIMD_FLOAT32_C(   696.63), EASYSIMD_FLOAT32_C(  -615.25), EASYSIMD_FLOAT32_C(  -729.71), EASYSIMD_FLOAT32_C(   450.87) },
       INT32_C(         193),
      { EASYSIMD_FLOAT32_C(  -478.91), EASYSIMD_FLOAT32_C(   219.09), EASYSIMD_FLOAT32_C(  -775.03), EASYSIMD_FLOAT32_C(  -972.69),
        EASYSIMD_FLOAT32_C(   696.63), EASYSIMD_FLOAT32_C(  -615.25), EASYSIMD_FLOAT32_C(  -729.71), EASYSIMD_FLOAT32_C(   450.87) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   281.93), EASYSIMD_FLOAT32_C(  -748.71), EASYSIMD_FLOAT32_C(   281.37),
        EASYSIMD_FLOAT32_C(   227.90), EASYSIMD_FLOAT32_C(   533.74), EASYSIMD_FLOAT32_C(   661.92), EASYSIMD_FLOAT32_C(  -668.36) },
       INT32_C(         209),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   281.93), EASYSIMD_FLOAT32_C(  -748.71), EASYSIMD_FLOAT32_C(   281.37),
        EASYSIMD_FLOAT32_C(   227.90), EASYSIMD_FLOAT32_C(   533.74), EASYSIMD_FLOAT32_C(   661.92), EASYSIMD_FLOAT32_C(  -668.36) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   397.08), EASYSIMD_FLOAT32_C(   420.11), EASYSIMD_FLOAT32_C(   -86.97),
        EASYSIMD_FLOAT32_C(  -505.02), EASYSIMD_FLOAT32_C(  -511.14), EASYSIMD_FLOAT32_C(  -589.74), EASYSIMD_FLOAT32_C(  -625.15) },
       INT32_C(         225),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   397.08), EASYSIMD_FLOAT32_C(   420.11), EASYSIMD_FLOAT32_C(   -86.97),
        EASYSIMD_FLOAT32_C(  -505.02), EASYSIMD_FLOAT32_C(  -511.14), EASYSIMD_FLOAT32_C(  -589.74), EASYSIMD_FLOAT32_C(  -625.15) } },
    { { EASYSIMD_FLOAT32_C(   762.65),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -235.14), EASYSIMD_FLOAT32_C(   -18.26),
        EASYSIMD_FLOAT32_C(   794.49), EASYSIMD_FLOAT32_C(  -207.83), EASYSIMD_FLOAT32_C(  -321.63), EASYSIMD_FLOAT32_C(  -820.76) },
       INT32_C(         241),
      { EASYSIMD_FLOAT32_C(   762.65),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -235.14), EASYSIMD_FLOAT32_C(   -18.26),
        EASYSIMD_FLOAT32_C(   794.49), EASYSIMD_FLOAT32_C(  -207.83), EASYSIMD_FLOAT32_C(  -321.63), EASYSIMD_FLOAT32_C(  -820.76) } },
    { { EASYSIMD_FLOAT32_C(   438.74),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   569.77), EASYSIMD_FLOAT32_C(  -279.89),
        EASYSIMD_FLOAT32_C(  -360.93), EASYSIMD_FLOAT32_C(   103.51), EASYSIMD_FLOAT32_C(  -617.97), EASYSIMD_FLOAT32_C(   -29.30) },
       INT32_C(           2),
      { EASYSIMD_FLOAT32_C(   439.00),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   570.00), EASYSIMD_FLOAT32_C(  -279.00),
        EASYSIMD_FLOAT32_C(  -360.00), EASYSIMD_FLOAT32_C(   104.00), EASYSIMD_FLOAT32_C(  -617.00), EASYSIMD_FLOAT32_C(   -29.00) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   698.37), EASYSIMD_FLOAT32_C(    59.36), EASYSIMD_FLOAT32_C(   -77.47),
        EASYSIMD_FLOAT32_C(  -812.77), EASYSIMD_FLOAT32_C(   469.62), EASYSIMD_FLOAT32_C(   297.37), EASYSIMD_FLOAT32_C(  -503.49) },
       INT32_C(          18),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   698.50), EASYSIMD_FLOAT32_C(    59.50), EASYSIMD_FLOAT32_C(   -77.00),
        EASYSIMD_FLOAT32_C(  -812.50), EASYSIMD_FLOAT32_C(   470.00), EASYSIMD_FLOAT32_C(   297.50), EASYSIMD_FLOAT32_C(  -503.00) } },
    { { EASYSIMD_FLOAT32_C(  -933.97),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -958.24), EASYSIMD_FLOAT32_C(   860.53),
        EASYSIMD_FLOAT32_C(   270.43), EASYSIMD_FLOAT32_C(  -279.88), EASYSIMD_FLOAT32_C(  -960.23), EASYSIMD_FLOAT32_C(  -667.11) },
       INT32_C(          34),
      { EASYSIMD_FLOAT32_C(  -933.75),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -958.00), EASYSIMD_FLOAT32_C(   860.75),
        EASYSIMD_FLOAT32_C(   270.50), EASYSIMD_FLOAT32_C(  -279.75), EASYSIMD_FLOAT32_C(  -960.00), EASYSIMD_FLOAT32_C(  -667.00) } },
    { { EASYSIMD_FLOAT32_C(  -739.47),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -508.26), EASYSIMD_FLOAT32_C(  -100.40),
        EASYSIMD_FLOAT32_C(  -968.46), EASYSIMD_FLOAT32_C(  -126.23), EASYSIMD_FLOAT32_C(   870.30), EASYSIMD_FLOAT32_C(    62.01) },
       INT32_C(          50),
      { EASYSIMD_FLOAT32_C(  -739.38),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -508.25), EASYSIMD_FLOAT32_C(  -100.38),
        EASYSIMD_FLOAT32_C(  -968.38), EASYSIMD_FLOAT32_C(  -126.12), EASYSIMD_FLOAT32_C(   870.38), EASYSIMD_FLOAT32_C(    62.12) } },
    { { EASYSIMD_FLOAT32_C(  -149.60), EASYSIMD_FLOAT32_C(    75.99), EASYSIMD_FLOAT32_C(  -587.92), EASYSIMD_FLOAT32_C(    37.62),
        EASYSIMD_FLOAT32_C(  -454.39), EASYSIMD_FLOAT32_C(   709.46), EASYSIMD_FLOAT32_C(   534.13), EASYSIMD_FLOAT32_C(  -741.00) },
       INT32_C(          66),
      { EASYSIMD_FLOAT32_C(  -149.56), EASYSIMD_FLOAT32_C(    76.00), EASYSIMD_FLOAT32_C(  -587.88), EASYSIMD_FLOAT32_C(    37.62),
        EASYSIMD_FLOAT32_C(  -454.38), EASYSIMD_FLOAT32_C(   709.50), EASYSIMD_FLOAT32_C(   534.19), EASYSIMD_FLOAT32_C(  -741.00) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -262.74), EASYSIMD_FLOAT32_C(  -188.77), EASYSIMD_FLOAT32_C(   460.69),
        EASYSIMD_FLOAT32_C(  -992.31), EASYSIMD_FLOAT32_C(   531.36), EASYSIMD_FLOAT32_C(   500.46), EASYSIMD_FLOAT32_C(  -659.43) },
       INT32_C(          82),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -262.72), EASYSIMD_FLOAT32_C(  -188.75), EASYSIMD_FLOAT32_C(   460.72),
        EASYSIMD_FLOAT32_C(  -992.28), EASYSIMD_FLOAT32_C(   531.38), EASYSIMD_FLOAT32_C(   500.47), EASYSIMD_FLOAT32_C(  -659.41) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -358.75), EASYSIMD_FLOAT32_C(   786.75), EASYSIMD_FLOAT32_C(  -396.06),
        EASYSIMD_FLOAT32_C(   540.85), EASYSIMD_FLOAT32_C(   818.29), EASYSIMD_FLOAT32_C(   477.71), EASYSIMD_FLOAT32_C(   411.15) },
       INT32_C(          98),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -358.75), EASYSIMD_FLOAT32_C(   786.75), EASYSIMD_FLOAT32_C(  -396.05),
        EASYSIMD_FLOAT32_C(   540.86), EASYSIMD_FLOAT32_C(   818.30), EASYSIMD_FLOAT32_C(   477.72), EASYSIMD_FLOAT32_C(   411.16) } },
    { { EASYSIMD_FLOAT32_C(   427.78), EASYSIMD_FLOAT32_C(  -630.14), EASYSIMD_FLOAT32_C(   480.13), EASYSIMD_FLOAT32_C(  -496.23),
        EASYSIMD_FLOAT32_C(  -218.06), EASYSIMD_FLOAT32_C(  -482.24), EASYSIMD_FLOAT32_C(    49.39), EASYSIMD_FLOAT32_C(  -508.60) },
       INT32_C(         114),
      { EASYSIMD_FLOAT32_C(   427.78), EASYSIMD_FLOAT32_C(  -630.13), EASYSIMD_FLOAT32_C(   480.13), EASYSIMD_FLOAT32_C(  -496.23),
        EASYSIMD_FLOAT32_C(  -218.05), EASYSIMD_FLOAT32_C(  -482.23), EASYSIMD_FLOAT32_C(    49.39), EASYSIMD_FLOAT32_C(  -508.59) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   260.88), EASYSIMD_FLOAT32_C(   652.05), EASYSIMD_FLOAT32_C(  -954.35),
        EASYSIMD_FLOAT32_C(  -927.89), EASYSIMD_FLOAT32_C(   112.74), EASYSIMD_FLOAT32_C(  -946.67), EASYSIMD_FLOAT32_C(   603.47) },
       INT32_C(         130),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   260.88), EASYSIMD_FLOAT32_C(   652.05), EASYSIMD_FLOAT32_C(  -954.35),
        EASYSIMD_FLOAT32_C(  -927.89), EASYSIMD_FLOAT32_C(   112.74), EASYSIMD_FLOAT32_C(  -946.67), EASYSIMD_FLOAT32_C(   603.47) } },
    { { EASYSIMD_FLOAT32_C(   984.19), EASYSIMD_FLOAT32_C(   471.92), EASYSIMD_FLOAT32_C(  -493.89), EASYSIMD_FLOAT32_C(  -374.56),
        EASYSIMD_FLOAT32_C(   258.67), EASYSIMD_FLOAT32_C(   110.05), EASYSIMD_FLOAT32_C(  -833.71), EASYSIMD_FLOAT32_C(    76.97) },
       INT32_C(         146),
      { EASYSIMD_FLOAT32_C(   984.19), EASYSIMD_FLOAT32_C(   471.92), EASYSIMD_FLOAT32_C(  -493.89), EASYSIMD_FLOAT32_C(  -374.56),
        EASYSIMD_FLOAT32_C(   258.67), EASYSIMD_FLOAT32_C(   110.05), EASYSIMD_FLOAT32_C(  -833.71), EASYSIMD_FLOAT32_C(    76.97) } },
    { { EASYSIMD_FLOAT32_C(   577.43),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   217.49), EASYSIMD_FLOAT32_C(     5.21),
        EASYSIMD_FLOAT32_C(  -672.88), EASYSIMD_FLOAT32_C(  -302.38), EASYSIMD_FLOAT32_C(   508.99), EASYSIMD_FLOAT32_C(   109.06) },
       INT32_C(         162),
      { EASYSIMD_FLOAT32_C(   577.43),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   217.49), EASYSIMD_FLOAT32_C(     5.21),
        EASYSIMD_FLOAT32_C(  -672.88), EASYSIMD_FLOAT32_C(  -302.38), EASYSIMD_FLOAT32_C(   508.99), EASYSIMD_FLOAT32_C(   109.06) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   866.76), EASYSIMD_FLOAT32_C(  -138.66), EASYSIMD_FLOAT32_C(   -80.68),
        EASYSIMD_FLOAT32_C(   912.41), EASYSIMD_FLOAT32_C(   -66.55), EASYSIMD_FLOAT32_C(  -967.94), EASYSIMD_FLOAT32_C(   965.74) },
       INT32_C(         178),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   866.76), EASYSIMD_FLOAT32_C(  -138.66), EASYSIMD_FLOAT32_C(   -80.68),
        EASYSIMD_FLOAT32_C(   912.41), EASYSIMD_FLOAT32_C(   -66.55), EASYSIMD_FLOAT32_C(  -967.94), EASYSIMD_FLOAT32_C(   965.74) } },
    { { EASYSIMD_FLOAT32_C(  -640.35),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -882.80), EASYSIMD_FLOAT32_C(  -134.25),
        EASYSIMD_FLOAT32_C(   146.55), EASYSIMD_FLOAT32_C(   375.87), EASYSIMD_FLOAT32_C(   975.80), EASYSIMD_FLOAT32_C(   312.84) },
       INT32_C(         194),
      { EASYSIMD_FLOAT32_C(  -640.35),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -882.80), EASYSIMD_FLOAT32_C(  -134.25),
        EASYSIMD_FLOAT32_C(   146.55), EASYSIMD_FLOAT32_C(   375.87), EASYSIMD_FLOAT32_C(   975.80), EASYSIMD_FLOAT32_C(   312.84) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   781.05), EASYSIMD_FLOAT32_C(   895.48), EASYSIMD_FLOAT32_C(  -262.78),
        EASYSIMD_FLOAT32_C(  -521.33), EASYSIMD_FLOAT32_C(   404.47), EASYSIMD_FLOAT32_C(   846.28), EASYSIMD_FLOAT32_C(   694.05) },
       INT32_C(         210),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   781.05), EASYSIMD_FLOAT32_C(   895.48), EASYSIMD_FLOAT32_C(  -262.78),
        EASYSIMD_FLOAT32_C(  -521.33), EASYSIMD_FLOAT32_C(   404.47), EASYSIMD_FLOAT32_C(   846.28), EASYSIMD_FLOAT32_C(   694.05) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   829.61), EASYSIMD_FLOAT32_C(  -691.92), EASYSIMD_FLOAT32_C(   880.64),
        EASYSIMD_FLOAT32_C(   742.02), EASYSIMD_FLOAT32_C(   241.53), EASYSIMD_FLOAT32_C(   912.71), EASYSIMD_FLOAT32_C(   707.76) },
       INT32_C(         226),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   829.61), EASYSIMD_FLOAT32_C(  -691.92), EASYSIMD_FLOAT32_C(   880.64),
        EASYSIMD_FLOAT32_C(   742.02), EASYSIMD_FLOAT32_C(   241.53), EASYSIMD_FLOAT32_C(   912.71), EASYSIMD_FLOAT32_C(   707.76) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -700.45), EASYSIMD_FLOAT32_C(  -324.83), EASYSIMD_FLOAT32_C(   -66.85),
        EASYSIMD_FLOAT32_C(   446.10), EASYSIMD_FLOAT32_C(  -948.96), EASYSIMD_FLOAT32_C(   -91.05), EASYSIMD_FLOAT32_C(  -241.06) },
       INT32_C(         242),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -700.45), EASYSIMD_FLOAT32_C(  -324.83), EASYSIMD_FLOAT32_C(   -66.85),
        EASYSIMD_FLOAT32_C(   446.10), EASYSIMD_FLOAT32_C(  -948.96), EASYSIMD_FLOAT32_C(   -91.05), EASYSIMD_FLOAT32_C(  -241.06) } },
    { { EASYSIMD_FLOAT32_C(   649.21), EASYSIMD_FLOAT32_C(   -86.03), EASYSIMD_FLOAT32_C(   253.56), EASYSIMD_FLOAT32_C(   544.69),
        EASYSIMD_FLOAT32_C(   651.19), EASYSIMD_FLOAT32_C(   732.23), EASYSIMD_FLOAT32_C(   -50.84), EASYSIMD_FLOAT32_C(   497.48) },
       INT32_C(           3),
      { EASYSIMD_FLOAT32_C(   649.00), EASYSIMD_FLOAT32_C(   -86.00), EASYSIMD_FLOAT32_C(   253.00), EASYSIMD_FLOAT32_C(   544.00),
        EASYSIMD_FLOAT32_C(   651.00), EASYSIMD_FLOAT32_C(   732.00), EASYSIMD_FLOAT32_C(   -50.00), EASYSIMD_FLOAT32_C(   497.00) } },
    { { EASYSIMD_FLOAT32_C(   -87.99), EASYSIMD_FLOAT32_C(   -55.78), EASYSIMD_FLOAT32_C(  -612.39), EASYSIMD_FLOAT32_C(  -258.38),
        EASYSIMD_FLOAT32_C(   252.30), EASYSIMD_FLOAT32_C(  -731.75), EASYSIMD_FLOAT32_C(  -516.36), EASYSIMD_FLOAT32_C(  -506.18) },
       INT32_C(          19),
      { EASYSIMD_FLOAT32_C(   -87.50), EASYSIMD_FLOAT32_C(   -55.50), EASYSIMD_FLOAT32_C(  -612.00), EASYSIMD_FLOAT32_C(  -258.00),
        EASYSIMD_FLOAT32_C(   252.00), EASYSIMD_FLOAT32_C(  -731.50), EASYSIMD_FLOAT32_C(  -516.00), EASYSIMD_FLOAT32_C(  -506.00) } },
    { { EASYSIMD_FLOAT32_C(  -808.61), EASYSIMD_FLOAT32_C(  -727.73), EASYSIMD_FLOAT32_C(  -261.07), EASYSIMD_FLOAT32_C(  -741.21),
        EASYSIMD_FLOAT32_C(  -428.18), EASYSIMD_FLOAT32_C(   414.11), EASYSIMD_FLOAT32_C(   191.95), EASYSIMD_FLOAT32_C(  -982.08) },
       INT32_C(          35),
      { EASYSIMD_FLOAT32_C(  -808.50), EASYSIMD_FLOAT32_C(  -727.50), EASYSIMD_FLOAT32_C(  -261.00), EASYSIMD_FLOAT32_C(  -741.00),
        EASYSIMD_FLOAT32_C(  -428.00), EASYSIMD_FLOAT32_C(   414.00), EASYSIMD_FLOAT32_C(   191.75), EASYSIMD_FLOAT32_C(  -982.00) } },
    { { EASYSIMD_FLOAT32_C(  -899.10),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   969.02), EASYSIMD_FLOAT32_C(   573.41),
        EASYSIMD_FLOAT32_C(  -573.94), EASYSIMD_FLOAT32_C(  -117.01), EASYSIMD_FLOAT32_C(  -173.03), EASYSIMD_FLOAT32_C(   970.75) },
       INT32_C(          51),
      { EASYSIMD_FLOAT32_C(  -899.00),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   969.00), EASYSIMD_FLOAT32_C(   573.38),
        EASYSIMD_FLOAT32_C(  -573.88), EASYSIMD_FLOAT32_C(  -117.00), EASYSIMD_FLOAT32_C(  -173.00), EASYSIMD_FLOAT32_C(   970.75) } },
    { { EASYSIMD_FLOAT32_C(  -968.33),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   831.93), EASYSIMD_FLOAT32_C(   -24.11),
        EASYSIMD_FLOAT32_C(  -626.91), EASYSIMD_FLOAT32_C(  -426.45), EASYSIMD_FLOAT32_C(  -771.82), EASYSIMD_FLOAT32_C(  -358.66) },
       INT32_C(          67),
      { EASYSIMD_FLOAT32_C(  -968.31),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   831.88), EASYSIMD_FLOAT32_C(   -24.06),
        EASYSIMD_FLOAT32_C(  -626.88), EASYSIMD_FLOAT32_C(  -426.44), EASYSIMD_FLOAT32_C(  -771.81), EASYSIMD_FLOAT32_C(  -358.62) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    -5.73), EASYSIMD_FLOAT32_C(   561.23), EASYSIMD_FLOAT32_C(   507.37),
        EASYSIMD_FLOAT32_C(   566.09), EASYSIMD_FLOAT32_C(   -24.66), EASYSIMD_FLOAT32_C(  -300.68), EASYSIMD_FLOAT32_C(   584.01) },
       INT32_C(          83),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    -5.72), EASYSIMD_FLOAT32_C(   561.22), EASYSIMD_FLOAT32_C(   507.34),
        EASYSIMD_FLOAT32_C(   566.06), EASYSIMD_FLOAT32_C(   -24.66), EASYSIMD_FLOAT32_C(  -300.66), EASYSIMD_FLOAT32_C(   584.00) } },
    { { EASYSIMD_FLOAT32_C(  -639.13), EASYSIMD_FLOAT32_C(  -590.50), EASYSIMD_FLOAT32_C(  -626.37), EASYSIMD_FLOAT32_C(  -213.07),
        EASYSIMD_FLOAT32_C(   292.50), EASYSIMD_FLOAT32_C(   200.60), EASYSIMD_FLOAT32_C(  -242.32), EASYSIMD_FLOAT32_C(   826.69) },
       INT32_C(          99),
      { EASYSIMD_FLOAT32_C(  -639.12), EASYSIMD_FLOAT32_C(  -590.50), EASYSIMD_FLOAT32_C(  -626.36), EASYSIMD_FLOAT32_C(  -213.06),
        EASYSIMD_FLOAT32_C(   292.50), EASYSIMD_FLOAT32_C(   200.59), EASYSIMD_FLOAT32_C(  -242.31), EASYSIMD_FLOAT32_C(   826.69) } },
    { { EASYSIMD_FLOAT32_C(   677.60),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   745.29), EASYSIMD_FLOAT32_C(   509.53),
        EASYSIMD_FLOAT32_C(  -165.76), EASYSIMD_FLOAT32_C(  -881.62), EASYSIMD_FLOAT32_C(  -916.92), EASYSIMD_FLOAT32_C(    62.43) },
       INT32_C(         115),
      { EASYSIMD_FLOAT32_C(   677.59),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   745.29), EASYSIMD_FLOAT32_C(   509.52),
        EASYSIMD_FLOAT32_C(  -165.76), EASYSIMD_FLOAT32_C(  -881.62), EASYSIMD_FLOAT32_C(  -916.91), EASYSIMD_FLOAT32_C(    62.43) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -611.15), EASYSIMD_FLOAT32_C(  -221.29), EASYSIMD_FLOAT32_C(   143.25),
        EASYSIMD_FLOAT32_C(   896.22), EASYSIMD_FLOAT32_C(  -655.20), EASYSIMD_FLOAT32_C(  -881.42), EASYSIMD_FLOAT32_C(  -404.46) },
       INT32_C(         131),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -611.15), EASYSIMD_FLOAT32_C(  -221.29), EASYSIMD_FLOAT32_C(   143.25),
        EASYSIMD_FLOAT32_C(   896.22), EASYSIMD_FLOAT32_C(  -655.20), EASYSIMD_FLOAT32_C(  -881.42), EASYSIMD_FLOAT32_C(  -404.46) } },
    { { EASYSIMD_FLOAT32_C(   395.76), EASYSIMD_FLOAT32_C(  -710.33), EASYSIMD_FLOAT32_C(   -31.43), EASYSIMD_FLOAT32_C(   769.40),
        EASYSIMD_FLOAT32_C(    76.60), EASYSIMD_FLOAT32_C(  -738.93), EASYSIMD_FLOAT32_C(   -30.00), EASYSIMD_FLOAT32_C(   834.28) },
       INT32_C(         147),
      { EASYSIMD_FLOAT32_C(   395.76), EASYSIMD_FLOAT32_C(  -710.33), EASYSIMD_FLOAT32_C(   -31.43), EASYSIMD_FLOAT32_C(   769.40),
        EASYSIMD_FLOAT32_C(    76.60), EASYSIMD_FLOAT32_C(  -738.93), EASYSIMD_FLOAT32_C(   -30.00), EASYSIMD_FLOAT32_C(   834.28) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   511.89), EASYSIMD_FLOAT32_C(   946.11), EASYSIMD_FLOAT32_C(  -524.91),
        EASYSIMD_FLOAT32_C(    21.42), EASYSIMD_FLOAT32_C(  -219.65), EASYSIMD_FLOAT32_C(  -406.54), EASYSIMD_FLOAT32_C(   104.50) },
       INT32_C(         163),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   511.89), EASYSIMD_FLOAT32_C(   946.11), EASYSIMD_FLOAT32_C(  -524.91),
        EASYSIMD_FLOAT32_C(    21.42), EASYSIMD_FLOAT32_C(  -219.65), EASYSIMD_FLOAT32_C(  -406.54), EASYSIMD_FLOAT32_C(   104.50) } },
    { { EASYSIMD_FLOAT32_C(  -755.23),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   -64.80), EASYSIMD_FLOAT32_C(  -366.38),
        EASYSIMD_FLOAT32_C(  -594.08), EASYSIMD_FLOAT32_C(  -921.55), EASYSIMD_FLOAT32_C(  -470.17), EASYSIMD_FLOAT32_C(  -249.28) },
       INT32_C(         179),
      { EASYSIMD_FLOAT32_C(  -755.23),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   -64.80), EASYSIMD_FLOAT32_C(  -366.38),
        EASYSIMD_FLOAT32_C(  -594.08), EASYSIMD_FLOAT32_C(  -921.55), EASYSIMD_FLOAT32_C(  -470.17), EASYSIMD_FLOAT32_C(  -249.28) } },
    { { EASYSIMD_FLOAT32_C(  -243.91), EASYSIMD_FLOAT32_C(  -478.87), EASYSIMD_FLOAT32_C(   -30.80), EASYSIMD_FLOAT32_C(   724.66),
        EASYSIMD_FLOAT32_C(  -709.47), EASYSIMD_FLOAT32_C(  -954.20), EASYSIMD_FLOAT32_C(   985.72), EASYSIMD_FLOAT32_C(   260.53) },
       INT32_C(         195),
      { EASYSIMD_FLOAT32_C(  -243.91), EASYSIMD_FLOAT32_C(  -478.87), EASYSIMD_FLOAT32_C(   -30.80), EASYSIMD_FLOAT32_C(   724.66),
        EASYSIMD_FLOAT32_C(  -709.47), EASYSIMD_FLOAT32_C(  -954.20), EASYSIMD_FLOAT32_C(   985.72), EASYSIMD_FLOAT32_C(   260.53) } },
    { { EASYSIMD_FLOAT32_C(  -926.52),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   391.97), EASYSIMD_FLOAT32_C(  -980.41),
        EASYSIMD_FLOAT32_C(  -534.59), EASYSIMD_FLOAT32_C(  -586.61), EASYSIMD_FLOAT32_C(  -200.06), EASYSIMD_FLOAT32_C(    58.88) },
       INT32_C(         211),
      { EASYSIMD_FLOAT32_C(  -926.52),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   391.97), EASYSIMD_FLOAT32_C(  -980.41),
        EASYSIMD_FLOAT32_C(  -534.59), EASYSIMD_FLOAT32_C(  -586.61), EASYSIMD_FLOAT32_C(  -200.06), EASYSIMD_FLOAT32_C(    58.88) } },
    { { EASYSIMD_FLOAT32_C(   762.66),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   347.26), EASYSIMD_FLOAT32_C(  -603.73),
        EASYSIMD_FLOAT32_C(  -324.15), EASYSIMD_FLOAT32_C(   425.71), EASYSIMD_FLOAT32_C(   -73.89), EASYSIMD_FLOAT32_C(   426.56) },
       INT32_C(         227),
      { EASYSIMD_FLOAT32_C(   762.66),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   347.26), EASYSIMD_FLOAT32_C(  -603.73),
        EASYSIMD_FLOAT32_C(  -324.15), EASYSIMD_FLOAT32_C(   425.71), EASYSIMD_FLOAT32_C(   -73.89), EASYSIMD_FLOAT32_C(   426.56) } },
    { { EASYSIMD_FLOAT32_C(  -621.18),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(    75.29), EASYSIMD_FLOAT32_C(  -896.52),
        EASYSIMD_FLOAT32_C(  -136.85), EASYSIMD_FLOAT32_C(   121.09), EASYSIMD_FLOAT32_C(  -910.79), EASYSIMD_FLOAT32_C(  -876.32) },
       INT32_C(         243),
      { EASYSIMD_FLOAT32_C(  -621.18),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(    75.29), EASYSIMD_FLOAT32_C(  -896.52),
        EASYSIMD_FLOAT32_C(  -136.85), EASYSIMD_FLOAT32_C(   121.09), EASYSIMD_FLOAT32_C(  -910.79), EASYSIMD_FLOAT32_C(  -876.32) } },
    { { EASYSIMD_FLOAT32_C(  -606.86), EASYSIMD_FLOAT32_C(  -817.73), EASYSIMD_FLOAT32_C(  -420.58), EASYSIMD_FLOAT32_C(  -193.47),
        EASYSIMD_FLOAT32_C(   -17.79), EASYSIMD_FLOAT32_C(   638.30), EASYSIMD_FLOAT32_C(  -675.58), EASYSIMD_FLOAT32_C(   624.93) },
       INT32_C(           4),
      { EASYSIMD_FLOAT32_C(  -607.00), EASYSIMD_FLOAT32_C(  -818.00), EASYSIMD_FLOAT32_C(  -421.00), EASYSIMD_FLOAT32_C(  -193.00),
        EASYSIMD_FLOAT32_C(   -18.00), EASYSIMD_FLOAT32_C(   638.00), EASYSIMD_FLOAT32_C(  -676.00), EASYSIMD_FLOAT32_C(   625.00) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   894.86), EASYSIMD_FLOAT32_C(   397.62), EASYSIMD_FLOAT32_C(  -516.64),
        EASYSIMD_FLOAT32_C(  -429.29), EASYSIMD_FLOAT32_C(  -176.67), EASYSIMD_FLOAT32_C(   409.46), EASYSIMD_FLOAT32_C(   997.28) },
       INT32_C(          20),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   895.00), EASYSIMD_FLOAT32_C(   397.50), EASYSIMD_FLOAT32_C(  -516.50),
        EASYSIMD_FLOAT32_C(  -429.50), EASYSIMD_FLOAT32_C(  -176.50), EASYSIMD_FLOAT32_C(   409.50), EASYSIMD_FLOAT32_C(   997.50) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -175.11), EASYSIMD_FLOAT32_C(  -966.44), EASYSIMD_FLOAT32_C(   178.65),
        EASYSIMD_FLOAT32_C(   -71.63), EASYSIMD_FLOAT32_C(  -103.29), EASYSIMD_FLOAT32_C(  -700.26), EASYSIMD_FLOAT32_C(    17.58) },
       INT32_C(          36),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -175.00), EASYSIMD_FLOAT32_C(  -966.50), EASYSIMD_FLOAT32_C(   178.75),
        EASYSIMD_FLOAT32_C(   -71.75), EASYSIMD_FLOAT32_C(  -103.25), EASYSIMD_FLOAT32_C(  -700.25), EASYSIMD_FLOAT32_C(    17.50) } },
    { { EASYSIMD_FLOAT32_C(   180.26), EASYSIMD_FLOAT32_C(   134.39), EASYSIMD_FLOAT32_C(   694.05), EASYSIMD_FLOAT32_C(   362.54),
        EASYSIMD_FLOAT32_C(   713.81), EASYSIMD_FLOAT32_C(  -499.42), EASYSIMD_FLOAT32_C(  -655.25), EASYSIMD_FLOAT32_C(   352.11) },
       INT32_C(          52),
      { EASYSIMD_FLOAT32_C(   180.25), EASYSIMD_FLOAT32_C(   134.38), EASYSIMD_FLOAT32_C(   694.00), EASYSIMD_FLOAT32_C(   362.50),
        EASYSIMD_FLOAT32_C(   713.75), EASYSIMD_FLOAT32_C(  -499.38), EASYSIMD_FLOAT32_C(  -655.25), EASYSIMD_FLOAT32_C(   352.12) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   402.47), EASYSIMD_FLOAT32_C(   -87.92), EASYSIMD_FLOAT32_C(   864.55),
        EASYSIMD_FLOAT32_C(  -199.91), EASYSIMD_FLOAT32_C(   395.44), EASYSIMD_FLOAT32_C(  -564.74), EASYSIMD_FLOAT32_C(   623.43) },
       INT32_C(          68),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   402.50), EASYSIMD_FLOAT32_C(   -87.94), EASYSIMD_FLOAT32_C(   864.56),
        EASYSIMD_FLOAT32_C(  -199.94), EASYSIMD_FLOAT32_C(   395.44), EASYSIMD_FLOAT32_C(  -564.75), EASYSIMD_FLOAT32_C(   623.44) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -734.15), EASYSIMD_FLOAT32_C(  -464.09), EASYSIMD_FLOAT32_C(  -105.62),
        EASYSIMD_FLOAT32_C(  -700.59), EASYSIMD_FLOAT32_C(   714.56), EASYSIMD_FLOAT32_C(   822.76), EASYSIMD_FLOAT32_C(   196.12) },
       INT32_C(          84),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -734.16), EASYSIMD_FLOAT32_C(  -464.09), EASYSIMD_FLOAT32_C(  -105.62),
        EASYSIMD_FLOAT32_C(  -700.59), EASYSIMD_FLOAT32_C(   714.56), EASYSIMD_FLOAT32_C(   822.75), EASYSIMD_FLOAT32_C(   196.12) } },
    { { EASYSIMD_FLOAT32_C(  -783.49),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -979.40), EASYSIMD_FLOAT32_C(   350.90),
        EASYSIMD_FLOAT32_C(     9.25), EASYSIMD_FLOAT32_C(   383.13), EASYSIMD_FLOAT32_C(    64.71), EASYSIMD_FLOAT32_C(   509.83) },
       INT32_C(         100),
      { EASYSIMD_FLOAT32_C(  -783.48),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -979.41), EASYSIMD_FLOAT32_C(   350.91),
        EASYSIMD_FLOAT32_C(     9.25), EASYSIMD_FLOAT32_C(   383.12), EASYSIMD_FLOAT32_C(    64.70), EASYSIMD_FLOAT32_C(   509.83) } },
    { { EASYSIMD_FLOAT32_C(   697.57), EASYSIMD_FLOAT32_C(   819.29), EASYSIMD_FLOAT32_C(   246.92), EASYSIMD_FLOAT32_C(   562.12),
        EASYSIMD_FLOAT32_C(  -380.62), EASYSIMD_FLOAT32_C(  -357.64), EASYSIMD_FLOAT32_C(   997.38), EASYSIMD_FLOAT32_C(  -757.19) },
       INT32_C(         116),
      { EASYSIMD_FLOAT32_C(   697.57), EASYSIMD_FLOAT32_C(   819.29), EASYSIMD_FLOAT32_C(   246.92), EASYSIMD_FLOAT32_C(   562.12),
        EASYSIMD_FLOAT32_C(  -380.62), EASYSIMD_FLOAT32_C(  -357.64), EASYSIMD_FLOAT32_C(   997.38), EASYSIMD_FLOAT32_C(  -757.19) } },
    { { EASYSIMD_FLOAT32_C(  -570.08), EASYSIMD_FLOAT32_C(  -687.70), EASYSIMD_FLOAT32_C(   713.11), EASYSIMD_FLOAT32_C(   -34.18),
        EASYSIMD_FLOAT32_C(   206.68), EASYSIMD_FLOAT32_C(  -987.48), EASYSIMD_FLOAT32_C(  -319.62), EASYSIMD_FLOAT32_C(    29.44) },
       INT32_C(         132),
      { EASYSIMD_FLOAT32_C(  -570.08), EASYSIMD_FLOAT32_C(  -687.70), EASYSIMD_FLOAT32_C(   713.11), EASYSIMD_FLOAT32_C(   -34.18),
        EASYSIMD_FLOAT32_C(   206.68), EASYSIMD_FLOAT32_C(  -987.48), EASYSIMD_FLOAT32_C(  -319.62), EASYSIMD_FLOAT32_C(    29.44) } },
    { { EASYSIMD_FLOAT32_C(  -305.32), EASYSIMD_FLOAT32_C(   869.78), EASYSIMD_FLOAT32_C(   425.14), EASYSIMD_FLOAT32_C(  -990.12),
        EASYSIMD_FLOAT32_C(   890.37), EASYSIMD_FLOAT32_C(  -223.96), EASYSIMD_FLOAT32_C(    19.13), EASYSIMD_FLOAT32_C(   273.51) },
       INT32_C(         148),
      { EASYSIMD_FLOAT32_C(  -305.32), EASYSIMD_FLOAT32_C(   869.78), EASYSIMD_FLOAT32_C(   425.14), EASYSIMD_FLOAT32_C(  -990.12),
        EASYSIMD_FLOAT32_C(   890.37), EASYSIMD_FLOAT32_C(  -223.96), EASYSIMD_FLOAT32_C(    19.13), EASYSIMD_FLOAT32_C(   273.51) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     1.39), EASYSIMD_FLOAT32_C(  -742.43), EASYSIMD_FLOAT32_C(  -136.20),
        EASYSIMD_FLOAT32_C(  -301.04), EASYSIMD_FLOAT32_C(  -923.15), EASYSIMD_FLOAT32_C(  -889.28), EASYSIMD_FLOAT32_C(  -738.92) },
       INT32_C(         164),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     1.39), EASYSIMD_FLOAT32_C(  -742.43), EASYSIMD_FLOAT32_C(  -136.20),
        EASYSIMD_FLOAT32_C(  -301.04), EASYSIMD_FLOAT32_C(  -923.15), EASYSIMD_FLOAT32_C(  -889.28), EASYSIMD_FLOAT32_C(  -738.92) } },
    { { EASYSIMD_FLOAT32_C(  -741.54), EASYSIMD_FLOAT32_C(   -60.96), EASYSIMD_FLOAT32_C(  -799.67), EASYSIMD_FLOAT32_C(  -311.62),
        EASYSIMD_FLOAT32_C(   251.34), EASYSIMD_FLOAT32_C(   913.44), EASYSIMD_FLOAT32_C(   654.20), EASYSIMD_FLOAT32_C(  -541.98) },
       INT32_C(         180),
      { EASYSIMD_FLOAT32_C(  -741.54), EASYSIMD_FLOAT32_C(   -60.96), EASYSIMD_FLOAT32_C(  -799.67), EASYSIMD_FLOAT32_C(  -311.62),
        EASYSIMD_FLOAT32_C(   251.34), EASYSIMD_FLOAT32_C(   913.44), EASYSIMD_FLOAT32_C(   654.20), EASYSIMD_FLOAT32_C(  -541.98) } },
    { { EASYSIMD_FLOAT32_C(  -665.42), EASYSIMD_FLOAT32_C(   487.46), EASYSIMD_FLOAT32_C(   134.59), EASYSIMD_FLOAT32_C(    29.26),
        EASYSIMD_FLOAT32_C(   357.24), EASYSIMD_FLOAT32_C(  -440.26), EASYSIMD_FLOAT32_C(    39.14), EASYSIMD_FLOAT32_C(   247.61) },
       INT32_C(         196),
      { EASYSIMD_FLOAT32_C(  -665.42), EASYSIMD_FLOAT32_C(   487.46), EASYSIMD_FLOAT32_C(   134.59), EASYSIMD_FLOAT32_C(    29.26),
        EASYSIMD_FLOAT32_C(   357.24), EASYSIMD_FLOAT32_C(  -440.26), EASYSIMD_FLOAT32_C(    39.14), EASYSIMD_FLOAT32_C(   247.61) } },
    { { EASYSIMD_FLOAT32_C(  -941.73),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   176.53), EASYSIMD_FLOAT32_C(  -412.77),
        EASYSIMD_FLOAT32_C(   522.52), EASYSIMD_FLOAT32_C(   434.09), EASYSIMD_FLOAT32_C(   451.03), EASYSIMD_FLOAT32_C(  -778.52) },
       INT32_C(         212),
      { EASYSIMD_FLOAT32_C(  -941.73),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   176.53), EASYSIMD_FLOAT32_C(  -412.77),
        EASYSIMD_FLOAT32_C(   522.52), EASYSIMD_FLOAT32_C(   434.09), EASYSIMD_FLOAT32_C(   451.03), EASYSIMD_FLOAT32_C(  -778.52) } },
    { { EASYSIMD_FLOAT32_C(  -792.82), EASYSIMD_FLOAT32_C(  -685.18), EASYSIMD_FLOAT32_C(  -258.98), EASYSIMD_FLOAT32_C(   146.22),
        EASYSIMD_FLOAT32_C(  -484.85), EASYSIMD_FLOAT32_C(   429.40), EASYSIMD_FLOAT32_C(  -602.44), EASYSIMD_FLOAT32_C(  -571.41) },
       INT32_C(         228),
      { EASYSIMD_FLOAT32_C(  -792.82), EASYSIMD_FLOAT32_C(  -685.18), EASYSIMD_FLOAT32_C(  -258.98), EASYSIMD_FLOAT32_C(   146.22),
        EASYSIMD_FLOAT32_C(  -484.85), EASYSIMD_FLOAT32_C(   429.40), EASYSIMD_FLOAT32_C(  -602.44), EASYSIMD_FLOAT32_C(  -571.41) } },
    { { EASYSIMD_FLOAT32_C(  -144.42), EASYSIMD_FLOAT32_C(  -645.45), EASYSIMD_FLOAT32_C(   418.18), EASYSIMD_FLOAT32_C(  -656.96),
        EASYSIMD_FLOAT32_C(   489.14), EASYSIMD_FLOAT32_C(  -552.57), EASYSIMD_FLOAT32_C(   700.28), EASYSIMD_FLOAT32_C(  -951.12) },
       INT32_C(         244),
      { EASYSIMD_FLOAT32_C(  -144.42), EASYSIMD_FLOAT32_C(  -645.45), EASYSIMD_FLOAT32_C(   418.18), EASYSIMD_FLOAT32_C(  -656.96),
        EASYSIMD_FLOAT32_C(   489.14), EASYSIMD_FLOAT32_C(  -552.57), EASYSIMD_FLOAT32_C(   700.28), EASYSIMD_FLOAT32_C(  -951.12) } },
  };

  easysimd__m256 a, r;

  a = easysimd_mm256_loadu_ps(test_vec[0].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(           0));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[1].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          16));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[2].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          32));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[3].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          48));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[4].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          64));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[5].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          80));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[6].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          96));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[7].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         112));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[7].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[8].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         128));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[8].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[9].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         144));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[9].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[10].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         160));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[10].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[11].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         176));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[11].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[12].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         192));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[12].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[13].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         208));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[13].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[14].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         224));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[14].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[15].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         240));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[15].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[16].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(           1));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[16].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[17].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          17));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[17].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[18].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          33));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[18].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[19].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          49));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[19].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[20].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          65));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[20].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[21].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          81));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[21].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[22].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          97));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[22].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[23].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         113));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[23].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[24].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         129));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[24].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[25].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         145));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[25].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[26].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         161));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[26].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[27].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         177));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[27].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[28].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         193));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[28].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[29].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         209));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[29].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[30].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         225));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[30].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[31].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         241));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[31].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[32].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(           2));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[32].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[33].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          18));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[33].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[34].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          34));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[34].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[35].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          50));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[35].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[36].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          66));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[36].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[37].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          82));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[37].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[38].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          98));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[38].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[39].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         114));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[39].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[40].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         130));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[40].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[41].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         146));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[41].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[42].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         162));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[42].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[43].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         178));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[43].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[44].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         194));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[44].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[45].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         210));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[45].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[46].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         226));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[46].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[47].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         242));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[47].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[48].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(           3));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[48].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[49].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          19));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[49].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[50].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          35));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[50].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[51].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          51));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[51].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[52].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          67));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[52].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[53].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          83));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[53].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[54].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          99));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[54].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[55].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         115));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[55].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[56].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         131));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[56].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[57].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         147));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[57].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[58].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         163));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[58].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[59].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         179));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[59].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[60].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         195));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[60].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[61].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         211));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[61].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[62].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         227));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[62].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[63].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         243));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[63].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[64].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(           4));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[64].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[65].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          20));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[65].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[66].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          36));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[66].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[67].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          52));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[67].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[68].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          68));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[68].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[69].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(          84));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[69].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[70].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         100));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[70].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[71].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         116));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[71].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[72].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         132));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[72].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[73].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         148));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[73].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[74].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         164));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[74].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[75].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         180));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[75].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[76].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         196));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[76].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[77].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         212));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[77].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[78].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         228));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[78].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[79].a);
  r = easysimd_mm256_roundscale_ps(a, INT32_C(         244));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[79].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 5 ; i++) {
    for (int j = 0 ; j < 16 ; j++) {
      easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
      if((easysimd_test_codegen_rand() & 1)) {
        if((easysimd_test_codegen_rand() & 1))
          a = easysimd_mm256_blend_ps(a, easysimd_mm256_set1_ps(EASYSIMD_MATH_NANF), 1);
        else {
          if((easysimd_test_codegen_rand() & 1))
            a = easysimd_mm256_blend_ps(a, easysimd_mm256_set1_ps(EASYSIMD_MATH_INFINITY), 2);
          else
            a = easysimd_mm256_blend_ps(a, easysimd_mm256_set1_ps(-EASYSIMD_MATH_INFINITY), 2);
        }
      }
      int imm8 = ((j << 4) | round_type[i]) & 255;
      easysimd__m256 r;
      EASYSIMD_CONSTIFY_256_(easysimd_mm256_roundscale_ps, r, easysimd_mm256_setzero_ps(), imm8, a);

      easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
      easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
      easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
    }
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_roundscale_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[8];
    const easysimd__mmask8 k;
    const easysimd_float32 a[8];
    const int32_t imm8;
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -382.29), EASYSIMD_FLOAT32_C(   330.96), EASYSIMD_FLOAT32_C(  -644.44), EASYSIMD_FLOAT32_C(  -895.86),
        EASYSIMD_FLOAT32_C(  -385.41), EASYSIMD_FLOAT32_C(    71.12), EASYSIMD_FLOAT32_C(  -739.06), EASYSIMD_FLOAT32_C(  -344.61) },
      UINT8_C(122),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -792.25), EASYSIMD_FLOAT32_C(  -224.49), EASYSIMD_FLOAT32_C(  -501.18),
        EASYSIMD_FLOAT32_C(  -757.05), EASYSIMD_FLOAT32_C(  -370.84), EASYSIMD_FLOAT32_C(   682.90), EASYSIMD_FLOAT32_C(  -710.21) },
       INT32_C(         128),
      { EASYSIMD_FLOAT32_C(  -382.29), EASYSIMD_FLOAT32_C(  -792.25), EASYSIMD_FLOAT32_C(  -644.44), EASYSIMD_FLOAT32_C(  -501.18),
        EASYSIMD_FLOAT32_C(  -757.05), EASYSIMD_FLOAT32_C(  -370.84), EASYSIMD_FLOAT32_C(   682.90), EASYSIMD_FLOAT32_C(  -344.61) } },
    { { EASYSIMD_FLOAT32_C(   148.79), EASYSIMD_FLOAT32_C(   905.05), EASYSIMD_FLOAT32_C(  -276.16), EASYSIMD_FLOAT32_C(   916.96),
        EASYSIMD_FLOAT32_C(  -272.05), EASYSIMD_FLOAT32_C(   859.28), EASYSIMD_FLOAT32_C(  -998.23), EASYSIMD_FLOAT32_C(  -263.23) },
      UINT8_C( 53),
      { EASYSIMD_FLOAT32_C(   936.67),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   162.47), EASYSIMD_FLOAT32_C(   267.63),
        EASYSIMD_FLOAT32_C(  -153.82), EASYSIMD_FLOAT32_C(   266.61), EASYSIMD_FLOAT32_C(   882.21), EASYSIMD_FLOAT32_C(   917.30) },
       INT32_C(         161),
      { EASYSIMD_FLOAT32_C(   936.67), EASYSIMD_FLOAT32_C(   905.05), EASYSIMD_FLOAT32_C(   162.47), EASYSIMD_FLOAT32_C(   916.96),
        EASYSIMD_FLOAT32_C(  -153.82), EASYSIMD_FLOAT32_C(   266.61), EASYSIMD_FLOAT32_C(  -998.23), EASYSIMD_FLOAT32_C(  -263.23) } },
    { { EASYSIMD_FLOAT32_C(  -254.64), EASYSIMD_FLOAT32_C(   772.87), EASYSIMD_FLOAT32_C(   717.70), EASYSIMD_FLOAT32_C(   -11.69),
        EASYSIMD_FLOAT32_C(  -597.96), EASYSIMD_FLOAT32_C(   400.60), EASYSIMD_FLOAT32_C(   278.10), EASYSIMD_FLOAT32_C(   809.76) },
      UINT8_C(221),
      { EASYSIMD_FLOAT32_C(    79.41), EASYSIMD_FLOAT32_C(   -41.45), EASYSIMD_FLOAT32_C(  -916.66), EASYSIMD_FLOAT32_C(   803.25),
        EASYSIMD_FLOAT32_C(  -124.49), EASYSIMD_FLOAT32_C(  -188.71), EASYSIMD_FLOAT32_C(   662.53), EASYSIMD_FLOAT32_C(  -122.72) },
       INT32_C(         130),
      { EASYSIMD_FLOAT32_C(    79.41), EASYSIMD_FLOAT32_C(   772.87), EASYSIMD_FLOAT32_C(  -916.66), EASYSIMD_FLOAT32_C(   803.25),
        EASYSIMD_FLOAT32_C(  -124.49), EASYSIMD_FLOAT32_C(   400.60), EASYSIMD_FLOAT32_C(   662.53), EASYSIMD_FLOAT32_C(  -122.72) } },
    { { EASYSIMD_FLOAT32_C(  -186.05), EASYSIMD_FLOAT32_C(  -961.32), EASYSIMD_FLOAT32_C(   369.76), EASYSIMD_FLOAT32_C(  -918.43),
        EASYSIMD_FLOAT32_C(  -115.14), EASYSIMD_FLOAT32_C(  -363.64), EASYSIMD_FLOAT32_C(   963.79), EASYSIMD_FLOAT32_C(  -197.84) },
      UINT8_C(146),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   799.52), EASYSIMD_FLOAT32_C(   382.78), EASYSIMD_FLOAT32_C(   246.75),
        EASYSIMD_FLOAT32_C(   572.39), EASYSIMD_FLOAT32_C(   100.49), EASYSIMD_FLOAT32_C(  -764.95), EASYSIMD_FLOAT32_C(   974.43) },
       INT32_C(         227),
      { EASYSIMD_FLOAT32_C(  -186.05), EASYSIMD_FLOAT32_C(   799.52), EASYSIMD_FLOAT32_C(   369.76), EASYSIMD_FLOAT32_C(  -918.43),
        EASYSIMD_FLOAT32_C(   572.39), EASYSIMD_FLOAT32_C(  -363.64), EASYSIMD_FLOAT32_C(   963.79), EASYSIMD_FLOAT32_C(   974.43) } },
    { { EASYSIMD_FLOAT32_C(  -320.62), EASYSIMD_FLOAT32_C(  -407.44), EASYSIMD_FLOAT32_C(  -257.26), EASYSIMD_FLOAT32_C(  -237.28),
        EASYSIMD_FLOAT32_C(  -604.19), EASYSIMD_FLOAT32_C(   618.25), EASYSIMD_FLOAT32_C(   574.00), EASYSIMD_FLOAT32_C(  -941.65) },
      UINT8_C(240),
      { EASYSIMD_FLOAT32_C(   122.06), EASYSIMD_FLOAT32_C(  -734.36), EASYSIMD_FLOAT32_C(   309.48), EASYSIMD_FLOAT32_C(   160.74),
        EASYSIMD_FLOAT32_C(   635.39), EASYSIMD_FLOAT32_C(   391.06), EASYSIMD_FLOAT32_C(  -954.40), EASYSIMD_FLOAT32_C(  -728.25) },
       INT32_C(         164),
      { EASYSIMD_FLOAT32_C(  -320.62), EASYSIMD_FLOAT32_C(  -407.44), EASYSIMD_FLOAT32_C(  -257.26), EASYSIMD_FLOAT32_C(  -237.28),
        EASYSIMD_FLOAT32_C(   635.39), EASYSIMD_FLOAT32_C(   391.06), EASYSIMD_FLOAT32_C(  -954.40), EASYSIMD_FLOAT32_C(  -728.25) } },
    { { EASYSIMD_FLOAT32_C(  -564.34), EASYSIMD_FLOAT32_C(   856.23), EASYSIMD_FLOAT32_C(  -352.72), EASYSIMD_FLOAT32_C(   818.44),
        EASYSIMD_FLOAT32_C(   102.98), EASYSIMD_FLOAT32_C(  -780.33), EASYSIMD_FLOAT32_C(   -81.07), EASYSIMD_FLOAT32_C(   338.03) },
      UINT8_C(186),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -148.82), EASYSIMD_FLOAT32_C(   978.30), EASYSIMD_FLOAT32_C(  -900.61),
        EASYSIMD_FLOAT32_C(   443.74), EASYSIMD_FLOAT32_C(  -278.96), EASYSIMD_FLOAT32_C(  -137.89), EASYSIMD_FLOAT32_C(   839.56) },
       INT32_C(         240),
      { EASYSIMD_FLOAT32_C(  -564.34), EASYSIMD_FLOAT32_C(  -148.82), EASYSIMD_FLOAT32_C(  -352.72), EASYSIMD_FLOAT32_C(  -900.61),
        EASYSIMD_FLOAT32_C(   443.74), EASYSIMD_FLOAT32_C(  -278.96), EASYSIMD_FLOAT32_C(   -81.07), EASYSIMD_FLOAT32_C(   839.56) } },
    { { EASYSIMD_FLOAT32_C(  -165.18), EASYSIMD_FLOAT32_C(   558.18), EASYSIMD_FLOAT32_C(  -836.46), EASYSIMD_FLOAT32_C(  -855.69),
        EASYSIMD_FLOAT32_C(  -281.08), EASYSIMD_FLOAT32_C(   798.94), EASYSIMD_FLOAT32_C(   535.36), EASYSIMD_FLOAT32_C(  -235.48) },
      UINT8_C(245),
      { EASYSIMD_FLOAT32_C(  -109.79), EASYSIMD_FLOAT32_C(   612.29), EASYSIMD_FLOAT32_C(  -493.65), EASYSIMD_FLOAT32_C(  -253.56),
        EASYSIMD_FLOAT32_C(  -740.43), EASYSIMD_FLOAT32_C(  -675.21), EASYSIMD_FLOAT32_C(   849.42), EASYSIMD_FLOAT32_C(  -520.76) },
       INT32_C(         161),
      { EASYSIMD_FLOAT32_C(  -109.79), EASYSIMD_FLOAT32_C(   558.18), EASYSIMD_FLOAT32_C(  -493.65), EASYSIMD_FLOAT32_C(  -855.69),
        EASYSIMD_FLOAT32_C(  -740.43), EASYSIMD_FLOAT32_C(  -675.21), EASYSIMD_FLOAT32_C(   849.42), EASYSIMD_FLOAT32_C(  -520.76) } },
    { { EASYSIMD_FLOAT32_C(  -326.65), EASYSIMD_FLOAT32_C(  -336.27), EASYSIMD_FLOAT32_C(  -961.37), EASYSIMD_FLOAT32_C(  -348.35),
        EASYSIMD_FLOAT32_C(  -236.87), EASYSIMD_FLOAT32_C(   482.37), EASYSIMD_FLOAT32_C(   372.69), EASYSIMD_FLOAT32_C(   625.24) },
      UINT8_C( 34),
      { EASYSIMD_FLOAT32_C(   711.98),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   219.84), EASYSIMD_FLOAT32_C(  -453.20),
        EASYSIMD_FLOAT32_C(   619.54), EASYSIMD_FLOAT32_C(   383.38), EASYSIMD_FLOAT32_C(  -308.89), EASYSIMD_FLOAT32_C(  -661.54) },
       INT32_C(           2),
      { EASYSIMD_FLOAT32_C(  -326.65),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -961.37), EASYSIMD_FLOAT32_C(  -348.35),
        EASYSIMD_FLOAT32_C(  -236.87), EASYSIMD_FLOAT32_C(   384.00), EASYSIMD_FLOAT32_C(   372.69), EASYSIMD_FLOAT32_C(   625.24) } },
  };

  easysimd__m256 src, a, r;

  src = easysimd_mm256_loadu_ps(test_vec[0].src);
  a = easysimd_mm256_loadu_ps(test_vec[0].a);
  r = easysimd_mm256_mask_roundscale_ps(src, test_vec[0].k, a, INT32_C(         128));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[0].r), 1);

  src = easysimd_mm256_loadu_ps(test_vec[1].src);
  a = easysimd_mm256_loadu_ps(test_vec[1].a);
  r = easysimd_mm256_mask_roundscale_ps(src, test_vec[1].k, a, INT32_C(         161));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[1].r), 1);

  src = easysimd_mm256_loadu_ps(test_vec[2].src);
  a = easysimd_mm256_loadu_ps(test_vec[2].a);
  r = easysimd_mm256_mask_roundscale_ps(src, test_vec[2].k, a, INT32_C(         130));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[2].r), 1);

  src = easysimd_mm256_loadu_ps(test_vec[3].src);
  a = easysimd_mm256_loadu_ps(test_vec[3].a);
  r = easysimd_mm256_mask_roundscale_ps(src, test_vec[3].k, a, INT32_C(         227));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[3].r), 1);

  src = easysimd_mm256_loadu_ps(test_vec[4].src);
  a = easysimd_mm256_loadu_ps(test_vec[4].a);
  r = easysimd_mm256_mask_roundscale_ps(src, test_vec[4].k, a, INT32_C(         164));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[4].r), 1);

  src = easysimd_mm256_loadu_ps(test_vec[5].src);
  a = easysimd_mm256_loadu_ps(test_vec[5].a);
  r = easysimd_mm256_mask_roundscale_ps(src, test_vec[5].k, a, INT32_C(         240));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[5].r), 1);

  src = easysimd_mm256_loadu_ps(test_vec[6].src);
  a = easysimd_mm256_loadu_ps(test_vec[6].a);
  r = easysimd_mm256_mask_roundscale_ps(src, test_vec[6].k, a, INT32_C(         161));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[6].r), 1);

  src = easysimd_mm256_loadu_ps(test_vec[7].src);
  a = easysimd_mm256_loadu_ps(test_vec[7].a);
  r = easysimd_mm256_mask_roundscale_ps(src, test_vec[7].k, a, INT32_C(           2));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 src = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    if((easysimd_test_codegen_rand() & 1)) {
      if((easysimd_test_codegen_rand() & 1))
        a = easysimd_mm256_blend_ps(a, easysimd_mm256_set1_ps(EASYSIMD_MATH_NANF), 1);
      else {
        if((easysimd_test_codegen_rand() & 1))
          a = easysimd_mm256_blend_ps(a, easysimd_mm256_set1_ps(EASYSIMD_MATH_INFINITY), 2);
        else
          a = easysimd_mm256_blend_ps(a, easysimd_mm256_set1_ps(-EASYSIMD_MATH_INFINITY), 2);
      }
    }
    int imm8 = (((easysimd_test_codegen_rand() & 15) << 4) | round_type[i % 5]) & 255;
    easysimd__m256 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm256_mask_roundscale_ps, r, easysimd_mm256_setzero_ps(), imm8, src, k, a);

    easysimd_test_x86_write_f32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_roundscale_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float32 a[8];
    const int32_t imm8;
    const easysimd_float32 r[8];
  } test_vec[] = {
    { UINT8_C(253),
      { EASYSIMD_FLOAT32_C(  -284.73), EASYSIMD_FLOAT32_C(   759.36), EASYSIMD_FLOAT32_C(   863.12), EASYSIMD_FLOAT32_C(   -25.16),
        EASYSIMD_FLOAT32_C(  -915.85), EASYSIMD_FLOAT32_C(   712.53), EASYSIMD_FLOAT32_C(   454.09), EASYSIMD_FLOAT32_C(   327.86) },
       INT32_C(           0),
      { EASYSIMD_FLOAT32_C(  -285.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   863.00), EASYSIMD_FLOAT32_C(   -25.00),
        EASYSIMD_FLOAT32_C(  -916.00), EASYSIMD_FLOAT32_C(   713.00), EASYSIMD_FLOAT32_C(   454.00), EASYSIMD_FLOAT32_C(   328.00) } },
    { UINT8_C(211),
      { EASYSIMD_FLOAT32_C(   -61.39), EASYSIMD_FLOAT32_C(  -220.92), EASYSIMD_FLOAT32_C(  -245.28), EASYSIMD_FLOAT32_C(  -579.01),
        EASYSIMD_FLOAT32_C(  -848.23), EASYSIMD_FLOAT32_C(  -620.04), EASYSIMD_FLOAT32_C(   742.92), EASYSIMD_FLOAT32_C(   863.75) },
       INT32_C(          65),
      { EASYSIMD_FLOAT32_C(   -61.44), EASYSIMD_FLOAT32_C(  -220.94), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  -848.25), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   742.88), EASYSIMD_FLOAT32_C(   863.75) } },
    { UINT8_C( 80),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -653.86), EASYSIMD_FLOAT32_C(   101.66), EASYSIMD_FLOAT32_C(  -600.69),
        EASYSIMD_FLOAT32_C(   528.46), EASYSIMD_FLOAT32_C(   328.14), EASYSIMD_FLOAT32_C(   502.30), EASYSIMD_FLOAT32_C(  -218.53) },
       INT32_C(         162),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   528.46), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   502.30), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 86),
      { EASYSIMD_FLOAT32_C(   192.41), EASYSIMD_FLOAT32_C(  -375.03), EASYSIMD_FLOAT32_C(  -979.53), EASYSIMD_FLOAT32_C(  -353.51),
        EASYSIMD_FLOAT32_C(   952.83), EASYSIMD_FLOAT32_C(   -79.55), EASYSIMD_FLOAT32_C(  -226.07), EASYSIMD_FLOAT32_C(   944.43) },
       INT32_C(           3),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -375.00), EASYSIMD_FLOAT32_C(  -979.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   952.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -226.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 65),
      { EASYSIMD_FLOAT32_C(  -719.95), EASYSIMD_FLOAT32_C(   704.78), EASYSIMD_FLOAT32_C(    79.10), EASYSIMD_FLOAT32_C(  -977.03),
        EASYSIMD_FLOAT32_C(   568.53), EASYSIMD_FLOAT32_C(   520.42), EASYSIMD_FLOAT32_C(   -14.27), EASYSIMD_FLOAT32_C(   979.08) },
       INT32_C(         244),
      { EASYSIMD_FLOAT32_C(  -719.95), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -14.27), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(149),
      { EASYSIMD_FLOAT32_C(   980.58),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -591.12), EASYSIMD_FLOAT32_C(   482.88),
        EASYSIMD_FLOAT32_C(   641.81), EASYSIMD_FLOAT32_C(  -146.31), EASYSIMD_FLOAT32_C(   700.45), EASYSIMD_FLOAT32_C(  -817.37) },
       INT32_C(          32),
      { EASYSIMD_FLOAT32_C(   980.50), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -591.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   641.75), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -817.25) } },
    { UINT8_C( 12),
      { EASYSIMD_FLOAT32_C(  -239.56), EASYSIMD_FLOAT32_C(  -897.46), EASYSIMD_FLOAT32_C(  -686.72), EASYSIMD_FLOAT32_C(  -295.14),
        EASYSIMD_FLOAT32_C(   961.60), EASYSIMD_FLOAT32_C(   866.29), EASYSIMD_FLOAT32_C(   404.01), EASYSIMD_FLOAT32_C(  -758.35) },
       INT32_C(         241),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -686.72), EASYSIMD_FLOAT32_C(  -295.14),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(176),
      { EASYSIMD_FLOAT32_C(   139.60),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   250.35), EASYSIMD_FLOAT32_C(   118.68),
        EASYSIMD_FLOAT32_C(   584.80), EASYSIMD_FLOAT32_C(  -417.77), EASYSIMD_FLOAT32_C(  -800.58), EASYSIMD_FLOAT32_C(   565.38) },
       INT32_C(          18),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   585.00), EASYSIMD_FLOAT32_C(  -417.50), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   565.50) } },
  };

  easysimd__m256 a, r;

  a = easysimd_mm256_loadu_ps(test_vec[0].a);
  r = easysimd_mm256_maskz_roundscale_ps(test_vec[0].k, a, INT32_C(           0));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[1].a);
  r = easysimd_mm256_maskz_roundscale_ps(test_vec[1].k, a, INT32_C(          65));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[2].a);
  r = easysimd_mm256_maskz_roundscale_ps(test_vec[2].k, a, INT32_C(         162));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[3].a);
  r = easysimd_mm256_maskz_roundscale_ps(test_vec[3].k, a, INT32_C(           3));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[4].a);
  r = easysimd_mm256_maskz_roundscale_ps(test_vec[4].k, a, INT32_C(         244));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[5].a);
  r = easysimd_mm256_maskz_roundscale_ps(test_vec[5].k, a, INT32_C(          32));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[6].a);
  r = easysimd_mm256_maskz_roundscale_ps(test_vec[6].k, a, INT32_C(         241));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[7].a);
  r = easysimd_mm256_maskz_roundscale_ps(test_vec[7].k, a, INT32_C(          18));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    if((easysimd_test_codegen_rand() & 1)) {
      if((easysimd_test_codegen_rand() & 1))
        a = easysimd_mm256_blend_ps(a, easysimd_mm256_set1_ps(EASYSIMD_MATH_NANF), 1);
      else {
        if((easysimd_test_codegen_rand() & 1))
          a = easysimd_mm256_blend_ps(a, easysimd_mm256_set1_ps(EASYSIMD_MATH_INFINITY), 2);
        else
          a = easysimd_mm256_blend_ps(a, easysimd_mm256_set1_ps(-EASYSIMD_MATH_INFINITY), 2);
      }
    }
    int imm8 = (((easysimd_test_codegen_rand() & 15) << 4) | round_type[i % 5]) & 255;
    easysimd__m256 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm256_maskz_roundscale_ps, r, easysimd_mm256_setzero_ps(), imm8, k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_roundscale_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[16];
    const int32_t imm8;
    const easysimd_float32 r[16];
  } test_vec[] = {
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -423.81), EASYSIMD_FLOAT32_C(  -634.18), EASYSIMD_FLOAT32_C(   554.60),
        EASYSIMD_FLOAT32_C(   173.43), EASYSIMD_FLOAT32_C(  -344.33), EASYSIMD_FLOAT32_C(  -362.67), EASYSIMD_FLOAT32_C(   218.91),
        EASYSIMD_FLOAT32_C(   598.01), EASYSIMD_FLOAT32_C(  -225.45), EASYSIMD_FLOAT32_C(   616.45), EASYSIMD_FLOAT32_C(  -416.23),
        EASYSIMD_FLOAT32_C(   663.74), EASYSIMD_FLOAT32_C(  -752.24), EASYSIMD_FLOAT32_C(  -575.44), EASYSIMD_FLOAT32_C(   644.36) },
       INT32_C(          64),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -423.81), EASYSIMD_FLOAT32_C(  -634.19), EASYSIMD_FLOAT32_C(   554.62),
        EASYSIMD_FLOAT32_C(   173.44), EASYSIMD_FLOAT32_C(  -344.31), EASYSIMD_FLOAT32_C(  -362.69), EASYSIMD_FLOAT32_C(   218.94),
        EASYSIMD_FLOAT32_C(   598.00), EASYSIMD_FLOAT32_C(  -225.44), EASYSIMD_FLOAT32_C(   616.44), EASYSIMD_FLOAT32_C(  -416.25),
        EASYSIMD_FLOAT32_C(   663.75), EASYSIMD_FLOAT32_C(  -752.25), EASYSIMD_FLOAT32_C(  -575.44), EASYSIMD_FLOAT32_C(   644.38) } },
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -162.62), EASYSIMD_FLOAT32_C(  -235.77), EASYSIMD_FLOAT32_C(  -694.67),
        EASYSIMD_FLOAT32_C(  -373.82), EASYSIMD_FLOAT32_C(   410.37), EASYSIMD_FLOAT32_C(  -543.77), EASYSIMD_FLOAT32_C(   -52.66),
        EASYSIMD_FLOAT32_C(   243.41), EASYSIMD_FLOAT32_C(   -55.65), EASYSIMD_FLOAT32_C(  -148.17), EASYSIMD_FLOAT32_C(   380.17),
        EASYSIMD_FLOAT32_C(   696.43), EASYSIMD_FLOAT32_C(   428.02), EASYSIMD_FLOAT32_C(   745.99), EASYSIMD_FLOAT32_C(   251.03) },
       INT32_C(           1),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -163.00), EASYSIMD_FLOAT32_C(  -236.00), EASYSIMD_FLOAT32_C(  -695.00),
        EASYSIMD_FLOAT32_C(  -374.00), EASYSIMD_FLOAT32_C(   410.00), EASYSIMD_FLOAT32_C(  -544.00), EASYSIMD_FLOAT32_C(   -53.00),
        EASYSIMD_FLOAT32_C(   243.00), EASYSIMD_FLOAT32_C(   -56.00), EASYSIMD_FLOAT32_C(  -149.00), EASYSIMD_FLOAT32_C(   380.00),
        EASYSIMD_FLOAT32_C(   696.00), EASYSIMD_FLOAT32_C(   428.00), EASYSIMD_FLOAT32_C(   745.00), EASYSIMD_FLOAT32_C(   251.00) } },
    { { EASYSIMD_FLOAT32_C(   820.36),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -337.09), EASYSIMD_FLOAT32_C(   436.80),
        EASYSIMD_FLOAT32_C(  -416.55), EASYSIMD_FLOAT32_C(  -673.34), EASYSIMD_FLOAT32_C(   684.56), EASYSIMD_FLOAT32_C(     8.00),
        EASYSIMD_FLOAT32_C(   971.02), EASYSIMD_FLOAT32_C(  -601.46), EASYSIMD_FLOAT32_C(  -889.67), EASYSIMD_FLOAT32_C(  -791.40),
        EASYSIMD_FLOAT32_C(  -347.68), EASYSIMD_FLOAT32_C(   -52.29), EASYSIMD_FLOAT32_C(   -27.17), EASYSIMD_FLOAT32_C(   -42.36) },
       INT32_C(          98),
      { EASYSIMD_FLOAT32_C(   820.38),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -337.08), EASYSIMD_FLOAT32_C(   436.81),
        EASYSIMD_FLOAT32_C(  -416.55), EASYSIMD_FLOAT32_C(  -673.33), EASYSIMD_FLOAT32_C(   684.56), EASYSIMD_FLOAT32_C(     8.00),
        EASYSIMD_FLOAT32_C(   971.03), EASYSIMD_FLOAT32_C(  -601.45), EASYSIMD_FLOAT32_C(  -889.66), EASYSIMD_FLOAT32_C(  -791.39),
        EASYSIMD_FLOAT32_C(  -347.67), EASYSIMD_FLOAT32_C(   -52.28), EASYSIMD_FLOAT32_C(   -27.16), EASYSIMD_FLOAT32_C(   -42.36) } },
    { { EASYSIMD_FLOAT32_C(   626.60), EASYSIMD_FLOAT32_C(  -641.77), EASYSIMD_FLOAT32_C(   373.06), EASYSIMD_FLOAT32_C(     6.77),
        EASYSIMD_FLOAT32_C(  -945.34), EASYSIMD_FLOAT32_C(  -198.92), EASYSIMD_FLOAT32_C(  -247.24), EASYSIMD_FLOAT32_C(   305.69),
        EASYSIMD_FLOAT32_C(   402.52), EASYSIMD_FLOAT32_C(   154.42), EASYSIMD_FLOAT32_C(   194.06), EASYSIMD_FLOAT32_C(   222.88),
        EASYSIMD_FLOAT32_C(   154.10), EASYSIMD_FLOAT32_C(   856.97), EASYSIMD_FLOAT32_C(  -340.32), EASYSIMD_FLOAT32_C(   737.54) },
       INT32_C(         243),
      { EASYSIMD_FLOAT32_C(   626.60), EASYSIMD_FLOAT32_C(  -641.77), EASYSIMD_FLOAT32_C(   373.06), EASYSIMD_FLOAT32_C(     6.77),
        EASYSIMD_FLOAT32_C(  -945.34), EASYSIMD_FLOAT32_C(  -198.92), EASYSIMD_FLOAT32_C(  -247.24), EASYSIMD_FLOAT32_C(   305.69),
        EASYSIMD_FLOAT32_C(   402.52), EASYSIMD_FLOAT32_C(   154.42), EASYSIMD_FLOAT32_C(   194.06), EASYSIMD_FLOAT32_C(   222.88),
        EASYSIMD_FLOAT32_C(   154.10), EASYSIMD_FLOAT32_C(   856.97), EASYSIMD_FLOAT32_C(  -340.32), EASYSIMD_FLOAT32_C(   737.54) } },
    { { EASYSIMD_FLOAT32_C(  -254.45), EASYSIMD_FLOAT32_C(  -845.35), EASYSIMD_FLOAT32_C(  -257.22), EASYSIMD_FLOAT32_C(  -144.12),
        EASYSIMD_FLOAT32_C(  -636.75), EASYSIMD_FLOAT32_C(   395.10), EASYSIMD_FLOAT32_C(   803.59), EASYSIMD_FLOAT32_C(   336.08),
        EASYSIMD_FLOAT32_C(  -647.26), EASYSIMD_FLOAT32_C(   377.48), EASYSIMD_FLOAT32_C(   719.28), EASYSIMD_FLOAT32_C(   766.61),
        EASYSIMD_FLOAT32_C(   898.72), EASYSIMD_FLOAT32_C(   345.88), EASYSIMD_FLOAT32_C(  -875.16), EASYSIMD_FLOAT32_C(   271.78) },
       INT32_C(         116),
      { EASYSIMD_FLOAT32_C(  -254.45), EASYSIMD_FLOAT32_C(  -845.35), EASYSIMD_FLOAT32_C(  -257.22), EASYSIMD_FLOAT32_C(  -144.12),
        EASYSIMD_FLOAT32_C(  -636.75), EASYSIMD_FLOAT32_C(   395.10), EASYSIMD_FLOAT32_C(   803.59), EASYSIMD_FLOAT32_C(   336.08),
        EASYSIMD_FLOAT32_C(  -647.26), EASYSIMD_FLOAT32_C(   377.48), EASYSIMD_FLOAT32_C(   719.28), EASYSIMD_FLOAT32_C(   766.61),
        EASYSIMD_FLOAT32_C(   898.72), EASYSIMD_FLOAT32_C(   345.88), EASYSIMD_FLOAT32_C(  -875.16), EASYSIMD_FLOAT32_C(   271.78) } },
    { { EASYSIMD_FLOAT32_C(  -927.14),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   485.19), EASYSIMD_FLOAT32_C(   475.38),
        EASYSIMD_FLOAT32_C(  -740.16), EASYSIMD_FLOAT32_C(  -320.75), EASYSIMD_FLOAT32_C(  -301.74), EASYSIMD_FLOAT32_C(   413.93),
        EASYSIMD_FLOAT32_C(  -463.77), EASYSIMD_FLOAT32_C(   357.95), EASYSIMD_FLOAT32_C(   151.48), EASYSIMD_FLOAT32_C(  -280.14),
        EASYSIMD_FLOAT32_C(   702.19), EASYSIMD_FLOAT32_C(   897.02), EASYSIMD_FLOAT32_C(  -125.49), EASYSIMD_FLOAT32_C(  -555.02) },
       INT32_C(          64),
      { EASYSIMD_FLOAT32_C(  -927.12),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   485.19), EASYSIMD_FLOAT32_C(   475.38),
        EASYSIMD_FLOAT32_C(  -740.19), EASYSIMD_FLOAT32_C(  -320.75), EASYSIMD_FLOAT32_C(  -301.75), EASYSIMD_FLOAT32_C(   413.94),
        EASYSIMD_FLOAT32_C(  -463.75), EASYSIMD_FLOAT32_C(   357.94), EASYSIMD_FLOAT32_C(   151.50), EASYSIMD_FLOAT32_C(  -280.12),
        EASYSIMD_FLOAT32_C(   702.19), EASYSIMD_FLOAT32_C(   897.00), EASYSIMD_FLOAT32_C(  -125.50), EASYSIMD_FLOAT32_C(  -555.00) } },
    { { EASYSIMD_FLOAT32_C(  -426.16),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   933.98), EASYSIMD_FLOAT32_C(  -706.88),
        EASYSIMD_FLOAT32_C(   959.43), EASYSIMD_FLOAT32_C(   832.70), EASYSIMD_FLOAT32_C(   639.00), EASYSIMD_FLOAT32_C(  -915.73),
        EASYSIMD_FLOAT32_C(   104.47), EASYSIMD_FLOAT32_C(   991.65), EASYSIMD_FLOAT32_C(  -736.23), EASYSIMD_FLOAT32_C(   177.33),
        EASYSIMD_FLOAT32_C(    97.07), EASYSIMD_FLOAT32_C(   748.97), EASYSIMD_FLOAT32_C(  -347.29), EASYSIMD_FLOAT32_C(   356.91) },
       INT32_C(         225),
      { EASYSIMD_FLOAT32_C(  -426.16),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   933.98), EASYSIMD_FLOAT32_C(  -706.88),
        EASYSIMD_FLOAT32_C(   959.43), EASYSIMD_FLOAT32_C(   832.70), EASYSIMD_FLOAT32_C(   639.00), EASYSIMD_FLOAT32_C(  -915.73),
        EASYSIMD_FLOAT32_C(   104.47), EASYSIMD_FLOAT32_C(   991.65), EASYSIMD_FLOAT32_C(  -736.23), EASYSIMD_FLOAT32_C(   177.33),
        EASYSIMD_FLOAT32_C(    97.07), EASYSIMD_FLOAT32_C(   748.97), EASYSIMD_FLOAT32_C(  -347.29), EASYSIMD_FLOAT32_C(   356.91) } },
    { { EASYSIMD_FLOAT32_C(  -291.08), EASYSIMD_FLOAT32_C(   922.32), EASYSIMD_FLOAT32_C(   684.31), EASYSIMD_FLOAT32_C(  -588.88),
        EASYSIMD_FLOAT32_C(   819.34), EASYSIMD_FLOAT32_C(  -441.18), EASYSIMD_FLOAT32_C(  -143.90), EASYSIMD_FLOAT32_C(  -427.75),
        EASYSIMD_FLOAT32_C(   796.58), EASYSIMD_FLOAT32_C(  -303.83), EASYSIMD_FLOAT32_C(   128.74), EASYSIMD_FLOAT32_C(  -629.58),
        EASYSIMD_FLOAT32_C(  -111.01), EASYSIMD_FLOAT32_C(    62.72), EASYSIMD_FLOAT32_C(  -336.46), EASYSIMD_FLOAT32_C(  -151.57) },
       INT32_C(          98),
      { EASYSIMD_FLOAT32_C(  -291.08), EASYSIMD_FLOAT32_C(   922.33), EASYSIMD_FLOAT32_C(   684.31), EASYSIMD_FLOAT32_C(  -588.88),
        EASYSIMD_FLOAT32_C(   819.34), EASYSIMD_FLOAT32_C(  -441.17), EASYSIMD_FLOAT32_C(  -143.89), EASYSIMD_FLOAT32_C(  -427.75),
        EASYSIMD_FLOAT32_C(   796.59), EASYSIMD_FLOAT32_C(  -303.83), EASYSIMD_FLOAT32_C(   128.75), EASYSIMD_FLOAT32_C(  -629.58),
        EASYSIMD_FLOAT32_C(  -111.00), EASYSIMD_FLOAT32_C(    62.73), EASYSIMD_FLOAT32_C(  -336.45), EASYSIMD_FLOAT32_C(  -151.56) } },
  };

  easysimd__m512 a, r;

  a = easysimd_mm512_loadu_ps(test_vec[0].a);
  r = easysimd_mm512_roundscale_ps(a, INT32_C(          64));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[1].a);
  r = easysimd_mm512_roundscale_ps(a, INT32_C(           1));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[2].a);
  r = easysimd_mm512_roundscale_ps(a, INT32_C(          98));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[3].a);
  r = easysimd_mm512_roundscale_ps(a, INT32_C(         243));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[4].a);
  r = easysimd_mm512_roundscale_ps(a, INT32_C(         116));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[5].a);
  r = easysimd_mm512_roundscale_ps(a, INT32_C(          64));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[6].a);
  r = easysimd_mm512_roundscale_ps(a, INT32_C(         225));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[7].a);
  r = easysimd_mm512_roundscale_ps(a, INT32_C(          98));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    if((easysimd_test_codegen_rand() & 1)) {
      if((easysimd_test_codegen_rand() & 1))
        a = easysimd_mm512_mask_mov_ps(a, 1, easysimd_mm512_set1_ps(EASYSIMD_MATH_NANF));
      else {
        if((easysimd_test_codegen_rand() & 1))
          a = easysimd_mm512_mask_mov_ps(a, 2, easysimd_mm512_set1_ps(EASYSIMD_MATH_INFINITY));
        else
          a = easysimd_mm512_mask_mov_ps(a, 2, easysimd_mm512_set1_ps(-EASYSIMD_MATH_INFINITY));
      }
    }
    int imm8 = (((easysimd_test_codegen_rand() & 15) << 4) | round_type[i % 5]) & 255;
    easysimd__m512 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm512_roundscale_ps, r, easysimd_mm512_setzero_ps(), imm8, a);

    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_roundscale_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[16];
    const easysimd__mmask8 k;
    const easysimd_float32 a[16];
    const int32_t imm8;
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -368.58), EASYSIMD_FLOAT32_C(   671.26), EASYSIMD_FLOAT32_C(   276.81), EASYSIMD_FLOAT32_C(   341.78),
        EASYSIMD_FLOAT32_C(   140.72), EASYSIMD_FLOAT32_C(   547.75), EASYSIMD_FLOAT32_C(   178.73), EASYSIMD_FLOAT32_C(   861.61),
        EASYSIMD_FLOAT32_C(    56.36), EASYSIMD_FLOAT32_C(  -214.56), EASYSIMD_FLOAT32_C(   152.99), EASYSIMD_FLOAT32_C(   284.00),
        EASYSIMD_FLOAT32_C(  -923.24), EASYSIMD_FLOAT32_C(  -638.78), EASYSIMD_FLOAT32_C(    48.36), EASYSIMD_FLOAT32_C(   907.10) },
      UINT8_C(140),
      { EASYSIMD_FLOAT32_C(  -196.81),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -866.78), EASYSIMD_FLOAT32_C(   796.60),
        EASYSIMD_FLOAT32_C(   299.16), EASYSIMD_FLOAT32_C(  -670.80), EASYSIMD_FLOAT32_C(  -931.45), EASYSIMD_FLOAT32_C(   327.80),
        EASYSIMD_FLOAT32_C(  -798.64), EASYSIMD_FLOAT32_C(  -170.99), EASYSIMD_FLOAT32_C(    72.32), EASYSIMD_FLOAT32_C(  -553.91),
        EASYSIMD_FLOAT32_C(   620.12), EASYSIMD_FLOAT32_C(   439.08), EASYSIMD_FLOAT32_C(    77.51), EASYSIMD_FLOAT32_C(   291.38) },
       INT32_C(          16),
      { EASYSIMD_FLOAT32_C(  -368.58), EASYSIMD_FLOAT32_C(   671.26), EASYSIMD_FLOAT32_C(  -867.00), EASYSIMD_FLOAT32_C(   796.50),
        EASYSIMD_FLOAT32_C(   140.72), EASYSIMD_FLOAT32_C(   547.75), EASYSIMD_FLOAT32_C(   178.73), EASYSIMD_FLOAT32_C(   328.00),
        EASYSIMD_FLOAT32_C(    56.36), EASYSIMD_FLOAT32_C(  -214.56), EASYSIMD_FLOAT32_C(   152.99), EASYSIMD_FLOAT32_C(   284.00),
        EASYSIMD_FLOAT32_C(  -923.24), EASYSIMD_FLOAT32_C(  -638.78), EASYSIMD_FLOAT32_C(    48.36), EASYSIMD_FLOAT32_C(   907.10) } },
    { { EASYSIMD_FLOAT32_C(   598.02), EASYSIMD_FLOAT32_C(  -706.29), EASYSIMD_FLOAT32_C(   320.00), EASYSIMD_FLOAT32_C(  -616.54),
        EASYSIMD_FLOAT32_C(   446.70), EASYSIMD_FLOAT32_C(  -396.00), EASYSIMD_FLOAT32_C(  -539.78), EASYSIMD_FLOAT32_C(   807.92),
        EASYSIMD_FLOAT32_C(   652.36), EASYSIMD_FLOAT32_C(  -632.67), EASYSIMD_FLOAT32_C(   781.30), EASYSIMD_FLOAT32_C(  -544.45),
        EASYSIMD_FLOAT32_C(  -180.13), EASYSIMD_FLOAT32_C(   914.53), EASYSIMD_FLOAT32_C(  -747.85), EASYSIMD_FLOAT32_C(  -880.97) },
      UINT8_C(192),
      { EASYSIMD_FLOAT32_C(  -679.30), EASYSIMD_FLOAT32_C(   446.83), EASYSIMD_FLOAT32_C(  -554.92), EASYSIMD_FLOAT32_C(   149.71),
        EASYSIMD_FLOAT32_C(  -480.86), EASYSIMD_FLOAT32_C(  -108.82), EASYSIMD_FLOAT32_C(  -230.17), EASYSIMD_FLOAT32_C(   958.22),
        EASYSIMD_FLOAT32_C(   968.69), EASYSIMD_FLOAT32_C(  -938.79), EASYSIMD_FLOAT32_C(  -325.89), EASYSIMD_FLOAT32_C(  -612.02),
        EASYSIMD_FLOAT32_C(  -506.69), EASYSIMD_FLOAT32_C(   -62.25), EASYSIMD_FLOAT32_C(   985.99), EASYSIMD_FLOAT32_C(  -212.98) },
       INT32_C(          33),
      { EASYSIMD_FLOAT32_C(   598.02), EASYSIMD_FLOAT32_C(  -706.29), EASYSIMD_FLOAT32_C(   320.00), EASYSIMD_FLOAT32_C(  -616.54),
        EASYSIMD_FLOAT32_C(   446.70), EASYSIMD_FLOAT32_C(  -396.00), EASYSIMD_FLOAT32_C(  -230.25), EASYSIMD_FLOAT32_C(   958.00),
        EASYSIMD_FLOAT32_C(   652.36), EASYSIMD_FLOAT32_C(  -632.67), EASYSIMD_FLOAT32_C(   781.30), EASYSIMD_FLOAT32_C(  -544.45),
        EASYSIMD_FLOAT32_C(  -180.13), EASYSIMD_FLOAT32_C(   914.53), EASYSIMD_FLOAT32_C(  -747.85), EASYSIMD_FLOAT32_C(  -880.97) } },
    { { EASYSIMD_FLOAT32_C(  -766.27), EASYSIMD_FLOAT32_C(  -138.25), EASYSIMD_FLOAT32_C(  -170.32), EASYSIMD_FLOAT32_C(  -958.35),
        EASYSIMD_FLOAT32_C(  -485.89), EASYSIMD_FLOAT32_C(   197.00), EASYSIMD_FLOAT32_C(   822.95), EASYSIMD_FLOAT32_C(   -30.33),
        EASYSIMD_FLOAT32_C(  -983.13), EASYSIMD_FLOAT32_C(   737.48), EASYSIMD_FLOAT32_C(   221.82), EASYSIMD_FLOAT32_C(  -864.10),
        EASYSIMD_FLOAT32_C(   981.21), EASYSIMD_FLOAT32_C(   542.52), EASYSIMD_FLOAT32_C(   582.73), EASYSIMD_FLOAT32_C(  -573.71) },
      UINT8_C(151),
      { EASYSIMD_FLOAT32_C(  -898.13),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   462.06), EASYSIMD_FLOAT32_C(  -939.91),
        EASYSIMD_FLOAT32_C(   286.15), EASYSIMD_FLOAT32_C(   523.26), EASYSIMD_FLOAT32_C(  -265.80), EASYSIMD_FLOAT32_C(   674.13),
        EASYSIMD_FLOAT32_C(  -983.43), EASYSIMD_FLOAT32_C(   671.95), EASYSIMD_FLOAT32_C(   660.12), EASYSIMD_FLOAT32_C(  -196.40),
        EASYSIMD_FLOAT32_C(   929.70), EASYSIMD_FLOAT32_C(  -970.42), EASYSIMD_FLOAT32_C(    37.32), EASYSIMD_FLOAT32_C(  -208.54) },
       INT32_C(          82),
      { EASYSIMD_FLOAT32_C(  -898.12),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   462.06), EASYSIMD_FLOAT32_C(  -958.35),
        EASYSIMD_FLOAT32_C(   286.16), EASYSIMD_FLOAT32_C(   197.00), EASYSIMD_FLOAT32_C(   822.95), EASYSIMD_FLOAT32_C(   674.16),
        EASYSIMD_FLOAT32_C(  -983.13), EASYSIMD_FLOAT32_C(   737.48), EASYSIMD_FLOAT32_C(   221.82), EASYSIMD_FLOAT32_C(  -864.10),
        EASYSIMD_FLOAT32_C(   981.21), EASYSIMD_FLOAT32_C(   542.52), EASYSIMD_FLOAT32_C(   582.73), EASYSIMD_FLOAT32_C(  -573.71) } },
    { { EASYSIMD_FLOAT32_C(   -98.07), EASYSIMD_FLOAT32_C(  -724.76), EASYSIMD_FLOAT32_C(  -926.87), EASYSIMD_FLOAT32_C(  -360.59),
        EASYSIMD_FLOAT32_C(   497.06), EASYSIMD_FLOAT32_C(  -790.97), EASYSIMD_FLOAT32_C(  -379.38), EASYSIMD_FLOAT32_C(    39.58),
        EASYSIMD_FLOAT32_C(   791.75), EASYSIMD_FLOAT32_C(    46.91), EASYSIMD_FLOAT32_C(   731.81), EASYSIMD_FLOAT32_C(   893.62),
        EASYSIMD_FLOAT32_C(  -635.63), EASYSIMD_FLOAT32_C(   193.87), EASYSIMD_FLOAT32_C(   953.71), EASYSIMD_FLOAT32_C(   650.53) },
      UINT8_C(118),
      { EASYSIMD_FLOAT32_C(  -312.08), EASYSIMD_FLOAT32_C(   324.66), EASYSIMD_FLOAT32_C(  -266.29), EASYSIMD_FLOAT32_C(  -640.13),
        EASYSIMD_FLOAT32_C(   -15.22), EASYSIMD_FLOAT32_C(   537.31), EASYSIMD_FLOAT32_C(  -710.43), EASYSIMD_FLOAT32_C(    14.36),
        EASYSIMD_FLOAT32_C(  -425.37), EASYSIMD_FLOAT32_C(    81.03), EASYSIMD_FLOAT32_C(   873.61), EASYSIMD_FLOAT32_C(   653.60),
        EASYSIMD_FLOAT32_C(  -613.39), EASYSIMD_FLOAT32_C(   929.87), EASYSIMD_FLOAT32_C(  -444.47), EASYSIMD_FLOAT32_C(  -338.15) },
       INT32_C(         211),
      { EASYSIMD_FLOAT32_C(   -98.07), EASYSIMD_FLOAT32_C(   324.66), EASYSIMD_FLOAT32_C(  -266.29), EASYSIMD_FLOAT32_C(  -360.59),
        EASYSIMD_FLOAT32_C(   -15.22), EASYSIMD_FLOAT32_C(   537.31), EASYSIMD_FLOAT32_C(  -710.43), EASYSIMD_FLOAT32_C(    39.58),
        EASYSIMD_FLOAT32_C(   791.75), EASYSIMD_FLOAT32_C(    46.91), EASYSIMD_FLOAT32_C(   731.81), EASYSIMD_FLOAT32_C(   893.62),
        EASYSIMD_FLOAT32_C(  -635.63), EASYSIMD_FLOAT32_C(   193.87), EASYSIMD_FLOAT32_C(   953.71), EASYSIMD_FLOAT32_C(   650.53) } },
    { { EASYSIMD_FLOAT32_C(  -841.09), EASYSIMD_FLOAT32_C(  -787.98), EASYSIMD_FLOAT32_C(   815.55), EASYSIMD_FLOAT32_C(   198.50),
        EASYSIMD_FLOAT32_C(  -996.22), EASYSIMD_FLOAT32_C(  -137.54), EASYSIMD_FLOAT32_C(   -69.69), EASYSIMD_FLOAT32_C(   897.40),
        EASYSIMD_FLOAT32_C(   226.83), EASYSIMD_FLOAT32_C(  -875.82), EASYSIMD_FLOAT32_C(   851.11), EASYSIMD_FLOAT32_C(  -122.64),
        EASYSIMD_FLOAT32_C(  -158.69), EASYSIMD_FLOAT32_C(  -460.97), EASYSIMD_FLOAT32_C(  -797.98), EASYSIMD_FLOAT32_C(   575.02) },
      UINT8_C( 79),
      { EASYSIMD_FLOAT32_C(   186.80), EASYSIMD_FLOAT32_C(   112.33), EASYSIMD_FLOAT32_C(   188.47), EASYSIMD_FLOAT32_C(  -798.85),
        EASYSIMD_FLOAT32_C(   686.96), EASYSIMD_FLOAT32_C(  -730.49), EASYSIMD_FLOAT32_C(  -925.24), EASYSIMD_FLOAT32_C(   340.56),
        EASYSIMD_FLOAT32_C(  -343.89), EASYSIMD_FLOAT32_C(  -995.37), EASYSIMD_FLOAT32_C(   896.09), EASYSIMD_FLOAT32_C(   317.96),
        EASYSIMD_FLOAT32_C(  -992.38), EASYSIMD_FLOAT32_C(    91.03), EASYSIMD_FLOAT32_C(   476.88), EASYSIMD_FLOAT32_C(  -780.35) },
       INT32_C(          84),
      { EASYSIMD_FLOAT32_C(   186.81), EASYSIMD_FLOAT32_C(   112.34), EASYSIMD_FLOAT32_C(   188.47), EASYSIMD_FLOAT32_C(  -798.84),
        EASYSIMD_FLOAT32_C(  -996.22), EASYSIMD_FLOAT32_C(  -137.54), EASYSIMD_FLOAT32_C(  -925.25), EASYSIMD_FLOAT32_C(   897.40),
        EASYSIMD_FLOAT32_C(   226.83), EASYSIMD_FLOAT32_C(  -875.82), EASYSIMD_FLOAT32_C(   851.11), EASYSIMD_FLOAT32_C(  -122.64),
        EASYSIMD_FLOAT32_C(  -158.69), EASYSIMD_FLOAT32_C(  -460.97), EASYSIMD_FLOAT32_C(  -797.98), EASYSIMD_FLOAT32_C(   575.02) } },
    { { EASYSIMD_FLOAT32_C(  -776.58), EASYSIMD_FLOAT32_C(   769.03), EASYSIMD_FLOAT32_C(   605.68), EASYSIMD_FLOAT32_C(  -879.18),
        EASYSIMD_FLOAT32_C(    -4.13), EASYSIMD_FLOAT32_C(   729.87), EASYSIMD_FLOAT32_C(   971.94), EASYSIMD_FLOAT32_C(   873.22),
        EASYSIMD_FLOAT32_C(  -428.82), EASYSIMD_FLOAT32_C(  -489.03), EASYSIMD_FLOAT32_C(  -924.76), EASYSIMD_FLOAT32_C(  -853.80),
        EASYSIMD_FLOAT32_C(   409.86), EASYSIMD_FLOAT32_C(   262.04), EASYSIMD_FLOAT32_C(   258.53), EASYSIMD_FLOAT32_C(  -401.66) },
      UINT8_C( 76),
      { EASYSIMD_FLOAT32_C(   -54.51), EASYSIMD_FLOAT32_C(  -132.16), EASYSIMD_FLOAT32_C(   537.95), EASYSIMD_FLOAT32_C(  -713.94),
        EASYSIMD_FLOAT32_C(   523.96), EASYSIMD_FLOAT32_C(   542.57), EASYSIMD_FLOAT32_C(  -817.85), EASYSIMD_FLOAT32_C(  -158.08),
        EASYSIMD_FLOAT32_C(   550.19), EASYSIMD_FLOAT32_C(   273.18), EASYSIMD_FLOAT32_C(  -681.20), EASYSIMD_FLOAT32_C(   769.84),
        EASYSIMD_FLOAT32_C(  -820.25), EASYSIMD_FLOAT32_C(    -5.83), EASYSIMD_FLOAT32_C(   993.26), EASYSIMD_FLOAT32_C(   948.79) },
       INT32_C(         144),
      { EASYSIMD_FLOAT32_C(  -776.58), EASYSIMD_FLOAT32_C(   769.03), EASYSIMD_FLOAT32_C(   537.95), EASYSIMD_FLOAT32_C(  -713.94),
        EASYSIMD_FLOAT32_C(    -4.13), EASYSIMD_FLOAT32_C(   729.87), EASYSIMD_FLOAT32_C(  -817.85), EASYSIMD_FLOAT32_C(   873.22),
        EASYSIMD_FLOAT32_C(  -428.82), EASYSIMD_FLOAT32_C(  -489.03), EASYSIMD_FLOAT32_C(  -924.76), EASYSIMD_FLOAT32_C(  -853.80),
        EASYSIMD_FLOAT32_C(   409.86), EASYSIMD_FLOAT32_C(   262.04), EASYSIMD_FLOAT32_C(   258.53), EASYSIMD_FLOAT32_C(  -401.66) } },
    { { EASYSIMD_FLOAT32_C(   -55.35), EASYSIMD_FLOAT32_C(  -670.28), EASYSIMD_FLOAT32_C(  -913.98), EASYSIMD_FLOAT32_C(  -182.13),
        EASYSIMD_FLOAT32_C(   -99.10), EASYSIMD_FLOAT32_C(  -403.01), EASYSIMD_FLOAT32_C(  -106.89), EASYSIMD_FLOAT32_C(    47.11),
        EASYSIMD_FLOAT32_C(  -993.15), EASYSIMD_FLOAT32_C(  -844.85), EASYSIMD_FLOAT32_C(  -694.36), EASYSIMD_FLOAT32_C(  -394.81),
        EASYSIMD_FLOAT32_C(   618.34), EASYSIMD_FLOAT32_C(   251.14), EASYSIMD_FLOAT32_C(   473.03), EASYSIMD_FLOAT32_C(   156.28) },
      UINT8_C( 92),
      { EASYSIMD_FLOAT32_C(    -3.01),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   719.34), EASYSIMD_FLOAT32_C(   838.91),
        EASYSIMD_FLOAT32_C(  -750.95), EASYSIMD_FLOAT32_C(    -7.48), EASYSIMD_FLOAT32_C(  -842.29), EASYSIMD_FLOAT32_C(  -981.11),
        EASYSIMD_FLOAT32_C(   172.27), EASYSIMD_FLOAT32_C(   151.88), EASYSIMD_FLOAT32_C(  -987.85), EASYSIMD_FLOAT32_C(   121.05),
        EASYSIMD_FLOAT32_C(   751.73), EASYSIMD_FLOAT32_C(  -873.77), EASYSIMD_FLOAT32_C(  -934.29), EASYSIMD_FLOAT32_C(  -918.54) },
       INT32_C(          17),
      { EASYSIMD_FLOAT32_C(   -55.35), EASYSIMD_FLOAT32_C(  -670.28), EASYSIMD_FLOAT32_C(   719.00), EASYSIMD_FLOAT32_C(   838.50),
        EASYSIMD_FLOAT32_C(  -751.00), EASYSIMD_FLOAT32_C(  -403.01), EASYSIMD_FLOAT32_C(  -842.50), EASYSIMD_FLOAT32_C(    47.11),
        EASYSIMD_FLOAT32_C(  -993.15), EASYSIMD_FLOAT32_C(  -844.85), EASYSIMD_FLOAT32_C(  -694.36), EASYSIMD_FLOAT32_C(  -394.81),
        EASYSIMD_FLOAT32_C(   618.34), EASYSIMD_FLOAT32_C(   251.14), EASYSIMD_FLOAT32_C(   473.03), EASYSIMD_FLOAT32_C(   156.28) } },
    { { EASYSIMD_FLOAT32_C(   776.69), EASYSIMD_FLOAT32_C(  -970.53), EASYSIMD_FLOAT32_C(  -183.90), EASYSIMD_FLOAT32_C(   931.85),
        EASYSIMD_FLOAT32_C(  -664.89), EASYSIMD_FLOAT32_C(   421.29), EASYSIMD_FLOAT32_C(   550.18), EASYSIMD_FLOAT32_C(   586.24),
        EASYSIMD_FLOAT32_C(  -105.68), EASYSIMD_FLOAT32_C(  -293.53), EASYSIMD_FLOAT32_C(   123.44), EASYSIMD_FLOAT32_C(   891.31),
        EASYSIMD_FLOAT32_C(   405.32), EASYSIMD_FLOAT32_C(  -157.22), EASYSIMD_FLOAT32_C(   730.22), EASYSIMD_FLOAT32_C(   654.37) },
      UINT8_C(159),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   673.25), EASYSIMD_FLOAT32_C(     7.57), EASYSIMD_FLOAT32_C(    39.80),
        EASYSIMD_FLOAT32_C(   685.40), EASYSIMD_FLOAT32_C(  -871.38), EASYSIMD_FLOAT32_C(  -208.47), EASYSIMD_FLOAT32_C(   811.64),
        EASYSIMD_FLOAT32_C(  -805.67), EASYSIMD_FLOAT32_C(  -127.01), EASYSIMD_FLOAT32_C(  -976.11), EASYSIMD_FLOAT32_C(    77.91),
        EASYSIMD_FLOAT32_C(   855.35), EASYSIMD_FLOAT32_C(  -166.86), EASYSIMD_FLOAT32_C(  -145.40), EASYSIMD_FLOAT32_C(   884.81) },
       INT32_C(         242),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   673.25), EASYSIMD_FLOAT32_C(     7.57), EASYSIMD_FLOAT32_C(    39.80),
        EASYSIMD_FLOAT32_C(   685.40), EASYSIMD_FLOAT32_C(   421.29), EASYSIMD_FLOAT32_C(   550.18), EASYSIMD_FLOAT32_C(   811.64),
        EASYSIMD_FLOAT32_C(  -105.68), EASYSIMD_FLOAT32_C(  -293.53), EASYSIMD_FLOAT32_C(   123.44), EASYSIMD_FLOAT32_C(   891.31),
        EASYSIMD_FLOAT32_C(   405.32), EASYSIMD_FLOAT32_C(  -157.22), EASYSIMD_FLOAT32_C(   730.22), EASYSIMD_FLOAT32_C(   654.37) } },
  };

  easysimd__m512 src, a, r;

  src = easysimd_mm512_loadu_ps(test_vec[0].src);
  a = easysimd_mm512_loadu_ps(test_vec[0].a);
  r = easysimd_mm512_mask_roundscale_ps(src, test_vec[0].k, a, INT32_C(          16));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[0].r), 1);

  src = easysimd_mm512_loadu_ps(test_vec[1].src);
  a = easysimd_mm512_loadu_ps(test_vec[1].a);
  r = easysimd_mm512_mask_roundscale_ps(src, test_vec[1].k, a, INT32_C(          33));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[1].r), 1);

  src = easysimd_mm512_loadu_ps(test_vec[2].src);
  a = easysimd_mm512_loadu_ps(test_vec[2].a);
  r = easysimd_mm512_mask_roundscale_ps(src, test_vec[2].k, a, INT32_C(          82));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[2].r), 1);

  src = easysimd_mm512_loadu_ps(test_vec[3].src);
  a = easysimd_mm512_loadu_ps(test_vec[3].a);
  r = easysimd_mm512_mask_roundscale_ps(src, test_vec[3].k, a, INT32_C(         211));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[3].r), 1);

  src = easysimd_mm512_loadu_ps(test_vec[4].src);
  a = easysimd_mm512_loadu_ps(test_vec[4].a);
  r = easysimd_mm512_mask_roundscale_ps(src, test_vec[4].k, a, INT32_C(          84));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[4].r), 1);

  src = easysimd_mm512_loadu_ps(test_vec[5].src);
  a = easysimd_mm512_loadu_ps(test_vec[5].a);
  r = easysimd_mm512_mask_roundscale_ps(src, test_vec[5].k, a, INT32_C(         144));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[5].r), 1);

  src = easysimd_mm512_loadu_ps(test_vec[6].src);
  a = easysimd_mm512_loadu_ps(test_vec[6].a);
  r = easysimd_mm512_mask_roundscale_ps(src, test_vec[6].k, a, INT32_C(          17));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[6].r), 1);

  src = easysimd_mm512_loadu_ps(test_vec[7].src);
  a = easysimd_mm512_loadu_ps(test_vec[7].a);
  r = easysimd_mm512_mask_roundscale_ps(src, test_vec[7].k, a, INT32_C(         242));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 src = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    if((easysimd_test_codegen_rand() & 1)) {
      if((easysimd_test_codegen_rand() & 1))
        a = easysimd_mm512_mask_mov_ps(a, 1, easysimd_mm512_set1_ps(EASYSIMD_MATH_NANF));
      else {
        if((easysimd_test_codegen_rand() & 1))
          a = easysimd_mm512_mask_mov_ps(a, 2, easysimd_mm512_set1_ps(EASYSIMD_MATH_INFINITY));
        else
          a = easysimd_mm512_mask_mov_ps(a, 2, easysimd_mm512_set1_ps(-EASYSIMD_MATH_INFINITY));
      }
    }
    int imm8 = (((easysimd_test_codegen_rand() & 15) << 4) | round_type[i % 5]) & 255;
    easysimd__m512 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm512_mask_roundscale_ps, r, easysimd_mm512_setzero_ps(), imm8, src, k, a);

    easysimd_test_x86_write_f32x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_roundscale_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float32 a[16];
    const int32_t imm8;
    const easysimd_float32 r[16];
  } test_vec[] = {
    { UINT8_C(234),
      { EASYSIMD_FLOAT32_C(   -13.26), EASYSIMD_FLOAT32_C(   198.00), EASYSIMD_FLOAT32_C(  -516.28), EASYSIMD_FLOAT32_C(  -586.06),
        EASYSIMD_FLOAT32_C(  -522.32), EASYSIMD_FLOAT32_C(  -834.30), EASYSIMD_FLOAT32_C(  -963.78), EASYSIMD_FLOAT32_C(   129.47),
        EASYSIMD_FLOAT32_C(   364.21), EASYSIMD_FLOAT32_C(   -20.29), EASYSIMD_FLOAT32_C(   293.22), EASYSIMD_FLOAT32_C(  -223.95),
        EASYSIMD_FLOAT32_C(  -730.48), EASYSIMD_FLOAT32_C(  -697.92), EASYSIMD_FLOAT32_C(   411.90), EASYSIMD_FLOAT32_C(  -489.41) },
       INT32_C(         208),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   198.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -586.06),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -834.30), EASYSIMD_FLOAT32_C(  -963.78), EASYSIMD_FLOAT32_C(   129.47),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 69),
      { EASYSIMD_FLOAT32_C(   771.32), EASYSIMD_FLOAT32_C(   547.03), EASYSIMD_FLOAT32_C(  -839.90), EASYSIMD_FLOAT32_C(  -855.36),
        EASYSIMD_FLOAT32_C(  -639.47), EASYSIMD_FLOAT32_C(  -711.44), EASYSIMD_FLOAT32_C(   257.81), EASYSIMD_FLOAT32_C(  -622.92),
        EASYSIMD_FLOAT32_C(    48.32), EASYSIMD_FLOAT32_C(    61.28), EASYSIMD_FLOAT32_C(  -328.47), EASYSIMD_FLOAT32_C(   892.68),
        EASYSIMD_FLOAT32_C(  -951.98), EASYSIMD_FLOAT32_C(   869.52), EASYSIMD_FLOAT32_C(  -623.59), EASYSIMD_FLOAT32_C(  -538.04) },
       INT32_C(         241),
      { EASYSIMD_FLOAT32_C(   771.32), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -839.90), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   257.81), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(219),
      { EASYSIMD_FLOAT32_C(   476.68),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   477.90), EASYSIMD_FLOAT32_C(  -230.10),
        EASYSIMD_FLOAT32_C(  -317.64), EASYSIMD_FLOAT32_C(   747.41), EASYSIMD_FLOAT32_C(    71.98), EASYSIMD_FLOAT32_C(  -905.73),
        EASYSIMD_FLOAT32_C(  -741.99), EASYSIMD_FLOAT32_C(  -985.21), EASYSIMD_FLOAT32_C(  -171.21), EASYSIMD_FLOAT32_C(   842.79),
        EASYSIMD_FLOAT32_C(   786.11), EASYSIMD_FLOAT32_C(  -624.19), EASYSIMD_FLOAT32_C(  -997.11), EASYSIMD_FLOAT32_C(   930.75) },
       INT32_C(         114),
      { EASYSIMD_FLOAT32_C(   476.69),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -230.09),
        EASYSIMD_FLOAT32_C(  -317.63), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    71.98), EASYSIMD_FLOAT32_C(  -905.73),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(247),
      { EASYSIMD_FLOAT32_C(  -750.15),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   232.46), EASYSIMD_FLOAT32_C(  -702.13),
        EASYSIMD_FLOAT32_C(   654.48), EASYSIMD_FLOAT32_C(   608.87), EASYSIMD_FLOAT32_C(  -240.17), EASYSIMD_FLOAT32_C(  -998.32),
        EASYSIMD_FLOAT32_C(  -849.02), EASYSIMD_FLOAT32_C(   258.02), EASYSIMD_FLOAT32_C(   478.36), EASYSIMD_FLOAT32_C(  -942.70),
        EASYSIMD_FLOAT32_C(  -264.09), EASYSIMD_FLOAT32_C(  -751.74), EASYSIMD_FLOAT32_C(  -260.33), EASYSIMD_FLOAT32_C(  -516.68) },
       INT32_C(          99),
      { EASYSIMD_FLOAT32_C(  -750.14),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   232.45), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   654.47), EASYSIMD_FLOAT32_C(   608.86), EASYSIMD_FLOAT32_C(  -240.16), EASYSIMD_FLOAT32_C(  -998.31),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(148),
      { EASYSIMD_FLOAT32_C(  -415.88), EASYSIMD_FLOAT32_C(   121.14), EASYSIMD_FLOAT32_C(  -961.46), EASYSIMD_FLOAT32_C(  -412.98),
        EASYSIMD_FLOAT32_C(    51.89), EASYSIMD_FLOAT32_C(  -225.11), EASYSIMD_FLOAT32_C(  -121.53), EASYSIMD_FLOAT32_C(  -759.54),
        EASYSIMD_FLOAT32_C(   888.31), EASYSIMD_FLOAT32_C(  -781.75), EASYSIMD_FLOAT32_C(  -509.69), EASYSIMD_FLOAT32_C(   673.26),
        EASYSIMD_FLOAT32_C(   450.72), EASYSIMD_FLOAT32_C(  -211.82), EASYSIMD_FLOAT32_C(   327.74), EASYSIMD_FLOAT32_C(    59.59) },
       INT32_C(         116),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -961.46), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(    51.89), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -759.54),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(  6),
      { EASYSIMD_FLOAT32_C(  -193.97), EASYSIMD_FLOAT32_C(  -192.22), EASYSIMD_FLOAT32_C(   267.87), EASYSIMD_FLOAT32_C(   541.94),
        EASYSIMD_FLOAT32_C(    56.04), EASYSIMD_FLOAT32_C(  -992.46), EASYSIMD_FLOAT32_C(  -974.74), EASYSIMD_FLOAT32_C(  -623.72),
        EASYSIMD_FLOAT32_C(  -158.53), EASYSIMD_FLOAT32_C(  -233.41), EASYSIMD_FLOAT32_C(   711.32), EASYSIMD_FLOAT32_C(  -495.81),
        EASYSIMD_FLOAT32_C(   350.72), EASYSIMD_FLOAT32_C(  -167.54), EASYSIMD_FLOAT32_C(  -457.27), EASYSIMD_FLOAT32_C(   937.73) },
       INT32_C(         128),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -192.22), EASYSIMD_FLOAT32_C(   267.87), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 28),
      { EASYSIMD_FLOAT32_C(  -875.18),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(    34.45), EASYSIMD_FLOAT32_C(  -384.88),
        EASYSIMD_FLOAT32_C(  -120.81), EASYSIMD_FLOAT32_C(  -514.83), EASYSIMD_FLOAT32_C(   403.30), EASYSIMD_FLOAT32_C(  -793.07),
        EASYSIMD_FLOAT32_C(   544.76), EASYSIMD_FLOAT32_C(   -48.68), EASYSIMD_FLOAT32_C(   536.35), EASYSIMD_FLOAT32_C(  -244.67),
        EASYSIMD_FLOAT32_C(   757.35), EASYSIMD_FLOAT32_C(  -655.86), EASYSIMD_FLOAT32_C(  -976.80), EASYSIMD_FLOAT32_C(   299.28) },
       INT32_C(          97),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    34.44), EASYSIMD_FLOAT32_C(  -384.89),
        EASYSIMD_FLOAT32_C(  -120.81), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 38),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   487.78), EASYSIMD_FLOAT32_C(   376.40), EASYSIMD_FLOAT32_C(   441.86),
        EASYSIMD_FLOAT32_C(  -679.76), EASYSIMD_FLOAT32_C(   919.14), EASYSIMD_FLOAT32_C(   379.59), EASYSIMD_FLOAT32_C(  -795.40),
        EASYSIMD_FLOAT32_C(   236.75), EASYSIMD_FLOAT32_C(  -804.21), EASYSIMD_FLOAT32_C(  -670.58), EASYSIMD_FLOAT32_C(  -557.32),
        EASYSIMD_FLOAT32_C(   230.25), EASYSIMD_FLOAT32_C(   -55.46), EASYSIMD_FLOAT32_C(   321.87), EASYSIMD_FLOAT32_C(   715.42) },
       INT32_C(         146),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   487.78), EASYSIMD_FLOAT32_C(   376.40), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   919.14), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
  };

  easysimd__m512 a, r;

  a = easysimd_mm512_loadu_ps(test_vec[0].a);
  r = easysimd_mm512_maskz_roundscale_ps(test_vec[0].k, a, INT32_C(         208));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[1].a);
  r = easysimd_mm512_maskz_roundscale_ps(test_vec[1].k, a, INT32_C(         241));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[2].a);
  r = easysimd_mm512_maskz_roundscale_ps(test_vec[2].k, a, INT32_C(         114));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[3].a);
  r = easysimd_mm512_maskz_roundscale_ps(test_vec[3].k, a, INT32_C(          99));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[4].a);
  r = easysimd_mm512_maskz_roundscale_ps(test_vec[4].k, a, INT32_C(         116));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[5].a);
  r = easysimd_mm512_maskz_roundscale_ps(test_vec[5].k, a, INT32_C(         128));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[6].a);
  r = easysimd_mm512_maskz_roundscale_ps(test_vec[6].k, a, INT32_C(          97));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[7].a);
  r = easysimd_mm512_maskz_roundscale_ps(test_vec[7].k, a, INT32_C(         146));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    if((easysimd_test_codegen_rand() & 1)) {
      if((easysimd_test_codegen_rand() & 1))
        a = easysimd_mm512_mask_mov_ps(a, 1, easysimd_mm512_set1_ps(EASYSIMD_MATH_NANF));
      else {
        if((easysimd_test_codegen_rand() & 1))
          a = easysimd_mm512_mask_mov_ps(a, 2, easysimd_mm512_set1_ps(EASYSIMD_MATH_INFINITY));
        else
          a = easysimd_mm512_mask_mov_ps(a, 2, easysimd_mm512_set1_ps(-EASYSIMD_MATH_INFINITY));
      }
    }
    int imm8 = (((easysimd_test_codegen_rand() & 15) << 4) | round_type[i % 5]) & 255;
    easysimd__m512 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm512_maskz_roundscale_ps, r, easysimd_mm512_setzero_ps(), imm8, k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_roundscale_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[2];
    const int imm8;
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   785.09), EASYSIMD_FLOAT64_C(  -944.62) },
       INT32_C(           0),
      { EASYSIMD_FLOAT64_C(   785.00), EASYSIMD_FLOAT64_C(  -945.00) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   918.83) },
       INT32_C(          16),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   919.00) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   618.81) },
       INT32_C(          32),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   618.75) } },
    { { EASYSIMD_FLOAT64_C(  -164.68),        EASYSIMD_MATH_INFINITY },
       INT32_C(          48),
      { EASYSIMD_FLOAT64_C(  -164.62),        EASYSIMD_MATH_INFINITY } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -996.38) },
       INT32_C(          64),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -996.38) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -175.51) },
       INT32_C(          80),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -175.50) } },
    { { EASYSIMD_FLOAT64_C(  -402.63), EASYSIMD_FLOAT64_C(   984.31) },
       INT32_C(          96),
      { EASYSIMD_FLOAT64_C(  -402.62), EASYSIMD_FLOAT64_C(   984.31) } },
    { { EASYSIMD_FLOAT64_C(  -217.00),        EASYSIMD_MATH_INFINITY },
       INT32_C(         112),
      { EASYSIMD_FLOAT64_C(  -217.00),        EASYSIMD_MATH_INFINITY } },
    { { EASYSIMD_FLOAT64_C(    86.63),       -EASYSIMD_MATH_INFINITY },
       INT32_C(         128),
      { EASYSIMD_FLOAT64_C(    86.63),       -EASYSIMD_MATH_INFINITY } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -856.07) },
       INT32_C(         144),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -856.07) } },
    { { EASYSIMD_FLOAT64_C(   795.64), EASYSIMD_FLOAT64_C(  -688.17) },
       INT32_C(         160),
      { EASYSIMD_FLOAT64_C(   795.64), EASYSIMD_FLOAT64_C(  -688.17) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(    64.12) },
       INT32_C(         176),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(    64.12) } },
    { { EASYSIMD_FLOAT64_C(    67.73), EASYSIMD_FLOAT64_C(   167.14) },
       INT32_C(         192),
      { EASYSIMD_FLOAT64_C(    67.73), EASYSIMD_FLOAT64_C(   167.14) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   991.63) },
       INT32_C(         208),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   991.63) } },
    { { EASYSIMD_FLOAT64_C(  -411.00),       -EASYSIMD_MATH_INFINITY },
       INT32_C(         224),
      { EASYSIMD_FLOAT64_C(  -411.00),       -EASYSIMD_MATH_INFINITY } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -456.55) },
       INT32_C(         240),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -456.55) } },
    { { EASYSIMD_FLOAT64_C(  -471.12), EASYSIMD_FLOAT64_C(  -358.30) },
       INT32_C(           1),
      { EASYSIMD_FLOAT64_C(  -472.00), EASYSIMD_FLOAT64_C(  -359.00) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -805.94) },
       INT32_C(          17),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -806.00) } },
    { { EASYSIMD_FLOAT64_C(   979.16),       -EASYSIMD_MATH_INFINITY },
       INT32_C(          33),
      { EASYSIMD_FLOAT64_C(   979.00),       -EASYSIMD_MATH_INFINITY } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   130.56) },
       INT32_C(          49),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   130.50) } },
    { { EASYSIMD_FLOAT64_C(  -702.31),       -EASYSIMD_MATH_INFINITY },
       INT32_C(          65),
      { EASYSIMD_FLOAT64_C(  -702.31),       -EASYSIMD_MATH_INFINITY } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -121.68) },
       INT32_C(          81),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -121.69) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   346.29) },
       INT32_C(          97),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   346.28) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(    41.41) },
       INT32_C(         113),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(    41.41) } },
    { { EASYSIMD_FLOAT64_C(  -907.56), EASYSIMD_FLOAT64_C(   709.24) },
       INT32_C(         129),
      { EASYSIMD_FLOAT64_C(  -907.56), EASYSIMD_FLOAT64_C(   709.24) } },
    { { EASYSIMD_FLOAT64_C(   287.39), EASYSIMD_FLOAT64_C(   572.34) },
       INT32_C(         145),
      { EASYSIMD_FLOAT64_C(   287.39), EASYSIMD_FLOAT64_C(   572.34) } },
    { { EASYSIMD_FLOAT64_C(  -722.01), EASYSIMD_FLOAT64_C(   747.27) },
       INT32_C(         161),
      { EASYSIMD_FLOAT64_C(  -722.01), EASYSIMD_FLOAT64_C(   747.27) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   -13.68) },
       INT32_C(         177),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   -13.68) } },
    { { EASYSIMD_FLOAT64_C(   293.11), EASYSIMD_FLOAT64_C(   430.09) },
       INT32_C(         193),
      { EASYSIMD_FLOAT64_C(   293.11), EASYSIMD_FLOAT64_C(   430.09) } },
    { { EASYSIMD_FLOAT64_C(    32.05), EASYSIMD_FLOAT64_C(   719.42) },
       INT32_C(         209),
      { EASYSIMD_FLOAT64_C(    32.05), EASYSIMD_FLOAT64_C(   719.42) } },
    { { EASYSIMD_FLOAT64_C(   141.27),       -EASYSIMD_MATH_INFINITY },
       INT32_C(         225),
      { EASYSIMD_FLOAT64_C(   141.27),       -EASYSIMD_MATH_INFINITY } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -825.97) },
       INT32_C(         241),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -825.97) } },
    { { EASYSIMD_FLOAT64_C(   215.44),        EASYSIMD_MATH_INFINITY },
       INT32_C(           2),
      { EASYSIMD_FLOAT64_C(   216.00),        EASYSIMD_MATH_INFINITY } },
    { { EASYSIMD_FLOAT64_C(   -89.17), EASYSIMD_FLOAT64_C(  -404.73) },
       INT32_C(          18),
      { EASYSIMD_FLOAT64_C(   -89.00), EASYSIMD_FLOAT64_C(  -404.50) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -126.74) },
       INT32_C(          34),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -126.50) } },
    { { EASYSIMD_FLOAT64_C(   722.04), EASYSIMD_FLOAT64_C(   -20.34) },
       INT32_C(          50),
      { EASYSIMD_FLOAT64_C(   722.12), EASYSIMD_FLOAT64_C(   -20.25) } },
    { { EASYSIMD_FLOAT64_C(  -225.17), EASYSIMD_FLOAT64_C(  -727.23) },
       INT32_C(          66),
      { EASYSIMD_FLOAT64_C(  -225.12), EASYSIMD_FLOAT64_C(  -727.19) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   304.82) },
       INT32_C(          82),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   304.84) } },
    { { EASYSIMD_FLOAT64_C(  -553.91), EASYSIMD_FLOAT64_C(   146.57) },
       INT32_C(          98),
      { EASYSIMD_FLOAT64_C(  -553.91), EASYSIMD_FLOAT64_C(   146.58) } },
    { { EASYSIMD_FLOAT64_C(   624.27), EASYSIMD_FLOAT64_C(   994.64) },
       INT32_C(         114),
      { EASYSIMD_FLOAT64_C(   624.27), EASYSIMD_FLOAT64_C(   994.64) } },
    { { EASYSIMD_FLOAT64_C(   798.30), EASYSIMD_FLOAT64_C(   636.47) },
       INT32_C(         130),
      { EASYSIMD_FLOAT64_C(   798.30), EASYSIMD_FLOAT64_C(   636.47) } },
    { { EASYSIMD_FLOAT64_C(    13.74),        EASYSIMD_MATH_INFINITY },
       INT32_C(         146),
      { EASYSIMD_FLOAT64_C(    13.74),        EASYSIMD_MATH_INFINITY } },
    { { EASYSIMD_FLOAT64_C(  -742.07), EASYSIMD_FLOAT64_C(   -83.10) },
       INT32_C(         162),
      { EASYSIMD_FLOAT64_C(  -742.07), EASYSIMD_FLOAT64_C(   -83.10) } },
    { { EASYSIMD_FLOAT64_C(  -476.98),       -EASYSIMD_MATH_INFINITY },
       INT32_C(         178),
      { EASYSIMD_FLOAT64_C(  -476.98),       -EASYSIMD_MATH_INFINITY } },
    { { EASYSIMD_FLOAT64_C(  -506.27),       -EASYSIMD_MATH_INFINITY },
       INT32_C(         194),
      { EASYSIMD_FLOAT64_C(  -506.27),       -EASYSIMD_MATH_INFINITY } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -928.68) },
       INT32_C(         210),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -928.68) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -285.92) },
       INT32_C(         226),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -285.92) } },
    { { EASYSIMD_FLOAT64_C(  -291.28), EASYSIMD_FLOAT64_C(   625.24) },
       INT32_C(         242),
      { EASYSIMD_FLOAT64_C(  -291.28), EASYSIMD_FLOAT64_C(   625.24) } },
    { { EASYSIMD_FLOAT64_C(  -654.81), EASYSIMD_FLOAT64_C(   436.59) },
       INT32_C(           3),
      { EASYSIMD_FLOAT64_C(  -654.00), EASYSIMD_FLOAT64_C(   436.00) } },
    { { EASYSIMD_FLOAT64_C(   -53.86), EASYSIMD_FLOAT64_C(  -216.31) },
       INT32_C(          19),
      { EASYSIMD_FLOAT64_C(   -53.50), EASYSIMD_FLOAT64_C(  -216.00) } },
    { { EASYSIMD_FLOAT64_C(  -779.18), EASYSIMD_FLOAT64_C(    41.62) },
       INT32_C(          35),
      { EASYSIMD_FLOAT64_C(  -779.00), EASYSIMD_FLOAT64_C(    41.50) } },
    { { EASYSIMD_FLOAT64_C(   741.56), EASYSIMD_FLOAT64_C(   564.64) },
       INT32_C(          51),
      { EASYSIMD_FLOAT64_C(   741.50), EASYSIMD_FLOAT64_C(   564.62) } },
    { { EASYSIMD_FLOAT64_C(   255.64), EASYSIMD_FLOAT64_C(  -645.42) },
       INT32_C(          67),
      { EASYSIMD_FLOAT64_C(   255.62), EASYSIMD_FLOAT64_C(  -645.38) } },
    { { EASYSIMD_FLOAT64_C(   749.37),       -EASYSIMD_MATH_INFINITY },
       INT32_C(          83),
      { EASYSIMD_FLOAT64_C(   749.34),       -EASYSIMD_MATH_INFINITY } },
    { { EASYSIMD_FLOAT64_C(   136.81),        EASYSIMD_MATH_INFINITY },
       INT32_C(          99),
      { EASYSIMD_FLOAT64_C(   136.80),        EASYSIMD_MATH_INFINITY } },
    { { EASYSIMD_FLOAT64_C(   844.12), EASYSIMD_FLOAT64_C(   832.76) },
       INT32_C(         115),
      { EASYSIMD_FLOAT64_C(   844.12), EASYSIMD_FLOAT64_C(   832.76) } },
    { { EASYSIMD_FLOAT64_C(  -447.16), EASYSIMD_FLOAT64_C(   458.00) },
       INT32_C(         131),
      { EASYSIMD_FLOAT64_C(  -447.16), EASYSIMD_FLOAT64_C(   458.00) } },
    { { EASYSIMD_FLOAT64_C(  -101.97), EASYSIMD_FLOAT64_C(  -105.41) },
       INT32_C(         147),
      { EASYSIMD_FLOAT64_C(  -101.97), EASYSIMD_FLOAT64_C(  -105.41) } },
    { { EASYSIMD_FLOAT64_C(   844.18), EASYSIMD_FLOAT64_C(   678.28) },
       INT32_C(         163),
      { EASYSIMD_FLOAT64_C(   844.18), EASYSIMD_FLOAT64_C(   678.28) } },
    { { EASYSIMD_FLOAT64_C(  -935.00), EASYSIMD_FLOAT64_C(  -280.09) },
       INT32_C(         179),
      { EASYSIMD_FLOAT64_C(  -935.00), EASYSIMD_FLOAT64_C(  -280.09) } },
    { { EASYSIMD_FLOAT64_C(   806.56), EASYSIMD_FLOAT64_C(  -715.45) },
       INT32_C(         195),
      { EASYSIMD_FLOAT64_C(   806.56), EASYSIMD_FLOAT64_C(  -715.45) } },
    { { EASYSIMD_FLOAT64_C(    62.19), EASYSIMD_FLOAT64_C(  -360.86) },
       INT32_C(         211),
      { EASYSIMD_FLOAT64_C(    62.19), EASYSIMD_FLOAT64_C(  -360.86) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   182.98) },
       INT32_C(         227),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   182.98) } },
    { { EASYSIMD_FLOAT64_C(  -254.49), EASYSIMD_FLOAT64_C(  -997.08) },
       INT32_C(         243),
      { EASYSIMD_FLOAT64_C(  -254.49), EASYSIMD_FLOAT64_C(  -997.08) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   667.08) },
       INT32_C(           4),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   667.00) } },
    { { EASYSIMD_FLOAT64_C(   499.84), EASYSIMD_FLOAT64_C(  -734.56) },
       INT32_C(          20),
      { EASYSIMD_FLOAT64_C(   500.00), EASYSIMD_FLOAT64_C(  -734.50) } },
    { { EASYSIMD_FLOAT64_C(   -42.15),        EASYSIMD_MATH_INFINITY },
       INT32_C(          36),
      { EASYSIMD_FLOAT64_C(   -42.25),        EASYSIMD_MATH_INFINITY } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   530.72) },
       INT32_C(          52),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   530.75) } },
    { { EASYSIMD_FLOAT64_C(  -749.37), EASYSIMD_FLOAT64_C(  -385.63) },
       INT32_C(          68),
      { EASYSIMD_FLOAT64_C(  -749.38), EASYSIMD_FLOAT64_C(  -385.62) } },
    { { EASYSIMD_FLOAT64_C(  -464.82), EASYSIMD_FLOAT64_C(  -795.73) },
       INT32_C(          84),
      { EASYSIMD_FLOAT64_C(  -464.81), EASYSIMD_FLOAT64_C(  -795.72) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   288.75) },
       INT32_C(         100),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   288.75) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -912.54) },
       INT32_C(         116),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -912.54) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   -21.64) },
       INT32_C(         132),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   -21.64) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   324.70) },
       INT32_C(         148),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   324.70) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -261.65) },
       INT32_C(         164),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -261.65) } },
    { { EASYSIMD_FLOAT64_C(  -670.05),       -EASYSIMD_MATH_INFINITY },
       INT32_C(         180),
      { EASYSIMD_FLOAT64_C(  -670.05),       -EASYSIMD_MATH_INFINITY } },
    { { EASYSIMD_FLOAT64_C(   916.34), EASYSIMD_FLOAT64_C(   951.18) },
       INT32_C(         196),
      { EASYSIMD_FLOAT64_C(   916.34), EASYSIMD_FLOAT64_C(   951.18) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -844.55) },
       INT32_C(         212),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -844.55) } },
    { { EASYSIMD_FLOAT64_C(   444.20), EASYSIMD_FLOAT64_C(   830.33) },
       INT32_C(         228),
      { EASYSIMD_FLOAT64_C(   444.20), EASYSIMD_FLOAT64_C(   830.33) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   917.79) },
       INT32_C(         244),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   917.79) } },
  };

  easysimd__m128d a, r;

  a = easysimd_mm_loadu_pd(test_vec[0].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(           0));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[1].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          16));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[2].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          32));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[3].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          48));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[4].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          64));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[5].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          80));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[6].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          96));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[7].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         112));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[7].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[8].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         128));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[8].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[9].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         144));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[9].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[10].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         160));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[10].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[11].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         176));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[11].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[12].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         192));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[12].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[13].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         208));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[13].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[14].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         224));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[14].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[15].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         240));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[15].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[16].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(           1));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[16].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[17].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          17));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[17].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[18].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          33));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[18].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[19].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          49));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[19].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[20].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          65));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[20].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[21].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          81));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[21].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[22].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          97));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[22].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[23].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         113));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[23].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[24].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         129));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[24].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[25].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         145));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[25].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[26].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         161));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[26].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[27].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         177));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[27].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[28].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         193));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[28].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[29].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         209));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[29].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[30].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         225));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[30].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[31].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         241));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[31].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[32].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(           2));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[32].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[33].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          18));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[33].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[34].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          34));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[34].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[35].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          50));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[35].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[36].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          66));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[36].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[37].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          82));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[37].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[38].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          98));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[38].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[39].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         114));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[39].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[40].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         130));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[40].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[41].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         146));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[41].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[42].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         162));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[42].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[43].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         178));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[43].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[44].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         194));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[44].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[45].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         210));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[45].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[46].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         226));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[46].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[47].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         242));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[47].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[48].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(           3));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[48].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[49].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          19));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[49].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[50].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          35));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[50].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[51].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          51));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[51].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[52].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          67));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[52].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[53].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          83));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[53].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[54].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          99));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[54].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[55].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         115));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[55].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[56].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         131));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[56].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[57].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         147));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[57].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[58].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         163));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[58].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[59].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         179));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[59].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[60].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         195));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[60].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[61].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         211));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[61].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[62].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         227));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[62].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[63].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         243));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[63].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[64].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(           4));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[64].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[65].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          20));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[65].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[66].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          36));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[66].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[67].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          52));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[67].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[68].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          68));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[68].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[69].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(          84));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[69].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[70].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         100));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[70].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[71].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         116));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[71].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[72].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         132));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[72].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[73].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         148));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[73].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[74].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         164));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[74].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[75].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         180));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[75].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[76].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         196));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[76].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[77].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         212));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[77].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[78].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         228));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[78].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[79].a);
  r = easysimd_mm_roundscale_pd(a, INT32_C(         244));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[79].r), 1);

 return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 5 ; i++) {
    for (int j = 0 ; j < 16 ; j++) {
      easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
      if((easysimd_test_codegen_rand() & 1)) {
        if((easysimd_test_codegen_rand() & 1))
          a = easysimd_mm_blend_pd(a, easysimd_mm_set1_pd(EASYSIMD_MATH_NAN), 1);
        else {
          if((easysimd_test_codegen_rand() & 1))
            a = easysimd_mm_blend_pd(a, easysimd_mm_set1_pd(EASYSIMD_MATH_INFINITY), 2);
          else
            a = easysimd_mm_blend_pd(a, easysimd_mm_set1_pd(-EASYSIMD_MATH_INFINITY), 2);
        }
      }
      int imm8 = ((j << 4) | round_type[i]) & 255;
      easysimd__m128d r;
      EASYSIMD_CONSTIFY_256_(easysimd_mm_roundscale_pd, r, easysimd_mm_setzero_pd(), imm8, a);

      easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
      easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
      easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
    }
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_roundscale_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[2];
    const easysimd__mmask8 k;
    const easysimd_float64 a[2];
    const int imm8;
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(    70.53), EASYSIMD_FLOAT64_C(  -663.37) },
      UINT8_C(228),
      { EASYSIMD_FLOAT64_C(   964.85), EASYSIMD_FLOAT64_C(    43.10) },
       INT32_C(         160),
      { EASYSIMD_FLOAT64_C(    70.53), EASYSIMD_FLOAT64_C(  -663.37) } },
    { { EASYSIMD_FLOAT64_C(  -551.58), EASYSIMD_FLOAT64_C(   772.38) },
      UINT8_C(206),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   607.68) },
       INT32_C(         209),
      { EASYSIMD_FLOAT64_C(  -551.58), EASYSIMD_FLOAT64_C(   607.68) } },
    { { EASYSIMD_FLOAT64_C(  -485.89), EASYSIMD_FLOAT64_C(   461.44) },
      UINT8_C(234),
      { EASYSIMD_FLOAT64_C(   305.64),        EASYSIMD_MATH_INFINITY },
       INT32_C(         210),
      { EASYSIMD_FLOAT64_C(  -485.89),        EASYSIMD_MATH_INFINITY } },
    { { EASYSIMD_FLOAT64_C(  -966.02), EASYSIMD_FLOAT64_C(  -869.88) },
      UINT8_C( 76),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   779.36) },
       INT32_C(         227),
      { EASYSIMD_FLOAT64_C(  -966.02), EASYSIMD_FLOAT64_C(  -869.88) } },
    { { EASYSIMD_FLOAT64_C(   993.79), EASYSIMD_FLOAT64_C(   944.88) },
      UINT8_C(110),
      { EASYSIMD_FLOAT64_C(    36.89), EASYSIMD_FLOAT64_C(  -125.52) },
       INT32_C(         212),
      { EASYSIMD_FLOAT64_C(   993.79), EASYSIMD_FLOAT64_C(  -125.52) } },
    { { EASYSIMD_FLOAT64_C(  -353.14), EASYSIMD_FLOAT64_C(  -742.71) },
      UINT8_C(176),
      { EASYSIMD_FLOAT64_C(  -745.46), EASYSIMD_FLOAT64_C(   731.60) },
       INT32_C(         224),
      { EASYSIMD_FLOAT64_C(  -353.14), EASYSIMD_FLOAT64_C(  -742.71) } },
    { { EASYSIMD_FLOAT64_C(  -754.30), EASYSIMD_FLOAT64_C(  -174.41) },
      UINT8_C(233),
      { EASYSIMD_FLOAT64_C(   551.34), EASYSIMD_FLOAT64_C(  -901.33) },
       INT32_C(         113),
      { EASYSIMD_FLOAT64_C(   551.34), EASYSIMD_FLOAT64_C(  -174.41) } },
    { { EASYSIMD_FLOAT64_C(   395.65), EASYSIMD_FLOAT64_C(  -432.03) },
      UINT8_C(122),
      { EASYSIMD_FLOAT64_C(   525.76), EASYSIMD_FLOAT64_C(   438.68) },
       INT32_C(          82),
      { EASYSIMD_FLOAT64_C(   395.65), EASYSIMD_FLOAT64_C(   438.69) } },
  };

  easysimd__m128d src, a, r;

  src = easysimd_mm_loadu_pd(test_vec[0].src);
  a = easysimd_mm_loadu_pd(test_vec[0].a);
  r = easysimd_mm_mask_roundscale_pd(src, test_vec[0].k, a, INT32_C(         160));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[0].r), 1);

  src = easysimd_mm_loadu_pd(test_vec[1].src);
  a = easysimd_mm_loadu_pd(test_vec[1].a);
  r = easysimd_mm_mask_roundscale_pd(src, test_vec[1].k, a, INT32_C(         209));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[1].r), 1);

  src = easysimd_mm_loadu_pd(test_vec[2].src);
  a = easysimd_mm_loadu_pd(test_vec[2].a);
  r = easysimd_mm_mask_roundscale_pd(src, test_vec[2].k, a, INT32_C(         210));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[2].r), 1);

  src = easysimd_mm_loadu_pd(test_vec[3].src);
  a = easysimd_mm_loadu_pd(test_vec[3].a);
  r = easysimd_mm_mask_roundscale_pd(src, test_vec[3].k, a, INT32_C(         227));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[3].r), 1);

  src = easysimd_mm_loadu_pd(test_vec[4].src);
  a = easysimd_mm_loadu_pd(test_vec[4].a);
  r = easysimd_mm_mask_roundscale_pd(src, test_vec[4].k, a, INT32_C(         212));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[4].r), 1);

  src = easysimd_mm_loadu_pd(test_vec[5].src);
  a = easysimd_mm_loadu_pd(test_vec[5].a);
  r = easysimd_mm_mask_roundscale_pd(src, test_vec[5].k, a, INT32_C(         224));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[5].r), 1);

  src = easysimd_mm_loadu_pd(test_vec[6].src);
  a = easysimd_mm_loadu_pd(test_vec[6].a);
  r = easysimd_mm_mask_roundscale_pd(src, test_vec[6].k, a, INT32_C(         113));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[6].r), 1);

  src = easysimd_mm_loadu_pd(test_vec[7].src);
  a = easysimd_mm_loadu_pd(test_vec[7].a);
  r = easysimd_mm_mask_roundscale_pd(src, test_vec[7].k, a, INT32_C(          82));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[7].r), 1);

 return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 8 ; i++) {
      easysimd__m128d src = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
      easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
      easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
      if((easysimd_test_codegen_rand() & 1)) {
        if((easysimd_test_codegen_rand() & 1))
          a = easysimd_mm_blend_pd(a, easysimd_mm_set1_pd(EASYSIMD_MATH_NAN), 1);
        else {
          if((easysimd_test_codegen_rand() & 1))
            a = easysimd_mm_blend_pd(a, easysimd_mm_set1_pd(EASYSIMD_MATH_INFINITY), 2);
          else
            a = easysimd_mm_blend_pd(a, easysimd_mm_set1_pd(-EASYSIMD_MATH_INFINITY), 2);
        }
      }
      int imm8 = (((easysimd_test_codegen_rand() & 15) << 4) | round_type[i % 5]) & 255;
      easysimd__m128d r;
      EASYSIMD_CONSTIFY_256_(easysimd_mm_mask_roundscale_pd, r, easysimd_mm_setzero_pd(), imm8, src, k, a);

      easysimd_test_x86_write_f64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
      easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
      easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
      easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
      easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_roundscale_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[2];
    const int imm8;
    const easysimd_float64 r[2];
  } test_vec[] = {
    { UINT8_C(129),
      { EASYSIMD_FLOAT64_C(   821.47), EASYSIMD_FLOAT64_C(  -844.99) },
       INT32_C(          96),
      { EASYSIMD_FLOAT64_C(   821.47), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 81),
      { EASYSIMD_FLOAT64_C(  -873.46), EASYSIMD_FLOAT64_C(  -359.17) },
       INT32_C(         193),
      { EASYSIMD_FLOAT64_C(  -873.46), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 63),
      { EASYSIMD_FLOAT64_C(   897.95),       -EASYSIMD_MATH_INFINITY },
       INT32_C(         242),
      { EASYSIMD_FLOAT64_C(   897.95),       -EASYSIMD_MATH_INFINITY } },
    { UINT8_C( 76),
      { EASYSIMD_FLOAT64_C(   389.69),       -EASYSIMD_MATH_INFINITY },
       INT32_C(          99),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(139),
      { EASYSIMD_FLOAT64_C(   145.51), EASYSIMD_FLOAT64_C(   -79.48) },
       INT32_C(         180),
      { EASYSIMD_FLOAT64_C(   145.51), EASYSIMD_FLOAT64_C(   -79.48) } },
    { UINT8_C(233),
      { EASYSIMD_FLOAT64_C(   714.89), EASYSIMD_FLOAT64_C(   680.04) },
       INT32_C(         128),
      { EASYSIMD_FLOAT64_C(   714.89), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(171),
      { EASYSIMD_FLOAT64_C(  -808.91), EASYSIMD_FLOAT64_C(  -160.34) },
       INT32_C(         241),
      { EASYSIMD_FLOAT64_C(  -808.91), EASYSIMD_FLOAT64_C(  -160.34) } },
    { UINT8_C( 19),
      { EASYSIMD_FLOAT64_C(  -491.94), EASYSIMD_FLOAT64_C(  -880.40) },
       INT32_C(          34),
      { EASYSIMD_FLOAT64_C(  -491.75), EASYSIMD_FLOAT64_C(  -880.25) } },
  };

  easysimd__m128d a, r;

  a = easysimd_mm_loadu_pd(test_vec[0].a);
  r = easysimd_mm_maskz_roundscale_pd(test_vec[0].k, a, INT32_C(          96));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[1].a);
  r = easysimd_mm_maskz_roundscale_pd(test_vec[1].k, a, INT32_C(         193));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[2].a);
  r = easysimd_mm_maskz_roundscale_pd(test_vec[2].k, a, INT32_C(         242));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[3].a);
  r = easysimd_mm_maskz_roundscale_pd(test_vec[3].k, a, INT32_C(          99));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[4].a);
  r = easysimd_mm_maskz_roundscale_pd(test_vec[4].k, a, INT32_C(         180));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[5].a);
  r = easysimd_mm_maskz_roundscale_pd(test_vec[5].k, a, INT32_C(         128));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[6].a);
  r = easysimd_mm_maskz_roundscale_pd(test_vec[6].k, a, INT32_C(         241));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[7].a);
  r = easysimd_mm_maskz_roundscale_pd(test_vec[7].k, a, INT32_C(          34));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[7].r), 1);

 return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    if((easysimd_test_codegen_rand() & 1)) {
      if((easysimd_test_codegen_rand() & 1))
        a = easysimd_mm_blend_pd(a, easysimd_mm_set1_pd(EASYSIMD_MATH_NAN), 1);
      else {
        if((easysimd_test_codegen_rand() & 1))
          a = easysimd_mm_blend_pd(a, easysimd_mm_set1_pd(EASYSIMD_MATH_INFINITY), 2);
        else
          a = easysimd_mm_blend_pd(a, easysimd_mm_set1_pd(-EASYSIMD_MATH_INFINITY), 2);
      }
    }
    int imm8 = (((easysimd_test_codegen_rand() & 15) << 4) | round_type[i % 5]) & 255;
    easysimd__m128d r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm_maskz_roundscale_pd, r, easysimd_mm_setzero_pd(), imm8, k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_roundscale_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[4];
    const int32_t imm8;
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   478.89),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -353.66), EASYSIMD_FLOAT64_C(   647.32) },
       INT32_C(           0),
      { EASYSIMD_FLOAT64_C(   479.00),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -354.00), EASYSIMD_FLOAT64_C(   647.00) } },
    { { EASYSIMD_FLOAT64_C(  -139.30), EASYSIMD_FLOAT64_C(  -585.30), EASYSIMD_FLOAT64_C(  -212.07), EASYSIMD_FLOAT64_C(  -727.58) },
       INT32_C(          16),
      { EASYSIMD_FLOAT64_C(  -139.50), EASYSIMD_FLOAT64_C(  -585.50), EASYSIMD_FLOAT64_C(  -212.00), EASYSIMD_FLOAT64_C(  -727.50) } },
    { { EASYSIMD_FLOAT64_C(  -424.71),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   300.82), EASYSIMD_FLOAT64_C(  -527.06) },
       INT32_C(          32),
      { EASYSIMD_FLOAT64_C(  -424.75),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   300.75), EASYSIMD_FLOAT64_C(  -527.00) } },
    { { EASYSIMD_FLOAT64_C(   759.84),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -784.82), EASYSIMD_FLOAT64_C(  -774.55) },
       INT32_C(          48),
      { EASYSIMD_FLOAT64_C(   759.88),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -784.88), EASYSIMD_FLOAT64_C(  -774.50) } },
    { { EASYSIMD_FLOAT64_C(    47.82),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   320.78), EASYSIMD_FLOAT64_C(  -567.27) },
       INT32_C(          64),
      { EASYSIMD_FLOAT64_C(    47.81),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   320.75), EASYSIMD_FLOAT64_C(  -567.25) } },
    { { EASYSIMD_FLOAT64_C(  -134.26),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   -18.89), EASYSIMD_FLOAT64_C(   416.85) },
       INT32_C(          80),
      { EASYSIMD_FLOAT64_C(  -134.25),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   -18.88), EASYSIMD_FLOAT64_C(   416.84) } },
    { { EASYSIMD_FLOAT64_C(   249.32), EASYSIMD_FLOAT64_C(  -885.77), EASYSIMD_FLOAT64_C(  -611.79), EASYSIMD_FLOAT64_C(   824.61) },
       INT32_C(          96),
      { EASYSIMD_FLOAT64_C(   249.31), EASYSIMD_FLOAT64_C(  -885.77), EASYSIMD_FLOAT64_C(  -611.80), EASYSIMD_FLOAT64_C(   824.61) } },
    { { EASYSIMD_FLOAT64_C(   689.03),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   135.80), EASYSIMD_FLOAT64_C(  -347.14) },
       INT32_C(         112),
      { EASYSIMD_FLOAT64_C(   689.03),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   135.80), EASYSIMD_FLOAT64_C(  -347.14) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   121.08), EASYSIMD_FLOAT64_C(   589.13), EASYSIMD_FLOAT64_C(   907.64) },
       INT32_C(         128),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   121.08), EASYSIMD_FLOAT64_C(   589.13), EASYSIMD_FLOAT64_C(   907.64) } },
    { { EASYSIMD_FLOAT64_C(   620.07),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(    69.68), EASYSIMD_FLOAT64_C(   839.47) },
       INT32_C(         144),
      { EASYSIMD_FLOAT64_C(   620.07),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(    69.68), EASYSIMD_FLOAT64_C(   839.47) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -196.65), EASYSIMD_FLOAT64_C(  -877.94), EASYSIMD_FLOAT64_C(  -295.05) },
       INT32_C(         160),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -196.65), EASYSIMD_FLOAT64_C(  -877.94), EASYSIMD_FLOAT64_C(  -295.05) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -240.61), EASYSIMD_FLOAT64_C(  -658.18), EASYSIMD_FLOAT64_C(   778.88) },
       INT32_C(         176),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -240.61), EASYSIMD_FLOAT64_C(  -658.18), EASYSIMD_FLOAT64_C(   778.88) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -992.29), EASYSIMD_FLOAT64_C(  -316.30), EASYSIMD_FLOAT64_C(  -447.51) },
       INT32_C(         192),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -992.29), EASYSIMD_FLOAT64_C(  -316.30), EASYSIMD_FLOAT64_C(  -447.51) } },
    { { EASYSIMD_FLOAT64_C(   243.73), EASYSIMD_FLOAT64_C(  -975.57), EASYSIMD_FLOAT64_C(   525.76), EASYSIMD_FLOAT64_C(   151.37) },
       INT32_C(         208),
      { EASYSIMD_FLOAT64_C(   243.73), EASYSIMD_FLOAT64_C(  -975.57), EASYSIMD_FLOAT64_C(   525.76), EASYSIMD_FLOAT64_C(   151.37) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -228.56), EASYSIMD_FLOAT64_C(  -302.52), EASYSIMD_FLOAT64_C(   232.39) },
       INT32_C(         224),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -228.56), EASYSIMD_FLOAT64_C(  -302.52), EASYSIMD_FLOAT64_C(   232.39) } },
    { { EASYSIMD_FLOAT64_C(    54.64),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -262.39), EASYSIMD_FLOAT64_C(   857.99) },
       INT32_C(         240),
      { EASYSIMD_FLOAT64_C(    54.64),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -262.39), EASYSIMD_FLOAT64_C(   857.99) } },
    { { EASYSIMD_FLOAT64_C(   391.80),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   262.54), EASYSIMD_FLOAT64_C(   733.62) },
       INT32_C(           1),
      { EASYSIMD_FLOAT64_C(   391.00),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   262.00), EASYSIMD_FLOAT64_C(   733.00) } },
    { { EASYSIMD_FLOAT64_C(   252.12), EASYSIMD_FLOAT64_C(  -857.84), EASYSIMD_FLOAT64_C(  -551.83), EASYSIMD_FLOAT64_C(   804.61) },
       INT32_C(          17),
      { EASYSIMD_FLOAT64_C(   252.00), EASYSIMD_FLOAT64_C(  -858.00), EASYSIMD_FLOAT64_C(  -552.00), EASYSIMD_FLOAT64_C(   804.50) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(    48.34), EASYSIMD_FLOAT64_C(    69.93), EASYSIMD_FLOAT64_C(   910.57) },
       INT32_C(          33),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(    48.25), EASYSIMD_FLOAT64_C(    69.75), EASYSIMD_FLOAT64_C(   910.50) } },
    { { EASYSIMD_FLOAT64_C(  -926.72),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   467.98), EASYSIMD_FLOAT64_C(   305.67) },
       INT32_C(          49),
      { EASYSIMD_FLOAT64_C(  -926.75),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   467.88), EASYSIMD_FLOAT64_C(   305.62) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -300.36), EASYSIMD_FLOAT64_C(  -781.70), EASYSIMD_FLOAT64_C(  -663.62) },
       INT32_C(          65),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -300.38), EASYSIMD_FLOAT64_C(  -781.75), EASYSIMD_FLOAT64_C(  -663.62) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   539.02), EASYSIMD_FLOAT64_C(   983.98), EASYSIMD_FLOAT64_C(   461.80) },
       INT32_C(          81),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   539.00), EASYSIMD_FLOAT64_C(   983.97), EASYSIMD_FLOAT64_C(   461.78) } },
    { { EASYSIMD_FLOAT64_C(   226.26), EASYSIMD_FLOAT64_C(   966.84), EASYSIMD_FLOAT64_C(  -739.41), EASYSIMD_FLOAT64_C(   674.43) },
       INT32_C(          97),
      { EASYSIMD_FLOAT64_C(   226.25), EASYSIMD_FLOAT64_C(   966.83), EASYSIMD_FLOAT64_C(  -739.42), EASYSIMD_FLOAT64_C(   674.42) } },
    { { EASYSIMD_FLOAT64_C(   306.09), EASYSIMD_FLOAT64_C(  -940.77), EASYSIMD_FLOAT64_C(  -180.22), EASYSIMD_FLOAT64_C(  -623.98) },
       INT32_C(         113),
      { EASYSIMD_FLOAT64_C(   306.09), EASYSIMD_FLOAT64_C(  -940.77), EASYSIMD_FLOAT64_C(  -180.23), EASYSIMD_FLOAT64_C(  -623.98) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   146.52), EASYSIMD_FLOAT64_C(  -956.93), EASYSIMD_FLOAT64_C(   990.64) },
       INT32_C(         129),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   146.52), EASYSIMD_FLOAT64_C(  -956.93), EASYSIMD_FLOAT64_C(   990.64) } },
    { { EASYSIMD_FLOAT64_C(   572.71), EASYSIMD_FLOAT64_C(  -423.47), EASYSIMD_FLOAT64_C(   709.06), EASYSIMD_FLOAT64_C(  -529.10) },
       INT32_C(         145),
      { EASYSIMD_FLOAT64_C(   572.71), EASYSIMD_FLOAT64_C(  -423.47), EASYSIMD_FLOAT64_C(   709.06), EASYSIMD_FLOAT64_C(  -529.10) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -192.72), EASYSIMD_FLOAT64_C(   418.37), EASYSIMD_FLOAT64_C(  -351.20) },
       INT32_C(         161),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -192.72), EASYSIMD_FLOAT64_C(   418.37), EASYSIMD_FLOAT64_C(  -351.20) } },
    { { EASYSIMD_FLOAT64_C(  -367.21),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   672.09), EASYSIMD_FLOAT64_C(  -248.78) },
       INT32_C(         177),
      { EASYSIMD_FLOAT64_C(  -367.21),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   672.09), EASYSIMD_FLOAT64_C(  -248.78) } },
    { { EASYSIMD_FLOAT64_C(  -102.05),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -682.10), EASYSIMD_FLOAT64_C(   -42.82) },
       INT32_C(         193),
      { EASYSIMD_FLOAT64_C(  -102.05),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -682.10), EASYSIMD_FLOAT64_C(   -42.82) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   840.45), EASYSIMD_FLOAT64_C(   -29.95), EASYSIMD_FLOAT64_C(   240.29) },
       INT32_C(         209),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   840.45), EASYSIMD_FLOAT64_C(   -29.95), EASYSIMD_FLOAT64_C(   240.29) } },
    { { EASYSIMD_FLOAT64_C(  -187.00), EASYSIMD_FLOAT64_C(    31.48), EASYSIMD_FLOAT64_C(  -972.15), EASYSIMD_FLOAT64_C(   283.90) },
       INT32_C(         225),
      { EASYSIMD_FLOAT64_C(  -187.00), EASYSIMD_FLOAT64_C(    31.48), EASYSIMD_FLOAT64_C(  -972.15), EASYSIMD_FLOAT64_C(   283.90) } },
    { { EASYSIMD_FLOAT64_C(   955.21), EASYSIMD_FLOAT64_C(  -908.82), EASYSIMD_FLOAT64_C(   726.01), EASYSIMD_FLOAT64_C(  -395.99) },
       INT32_C(         241),
      { EASYSIMD_FLOAT64_C(   955.21), EASYSIMD_FLOAT64_C(  -908.82), EASYSIMD_FLOAT64_C(   726.01), EASYSIMD_FLOAT64_C(  -395.99) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   236.80), EASYSIMD_FLOAT64_C(  -376.09), EASYSIMD_FLOAT64_C(  -644.51) },
       INT32_C(           2),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   237.00), EASYSIMD_FLOAT64_C(  -376.00), EASYSIMD_FLOAT64_C(  -644.00) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(    -0.17), EASYSIMD_FLOAT64_C(  -254.61), EASYSIMD_FLOAT64_C(   404.79) },
       INT32_C(          18),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(  -254.50), EASYSIMD_FLOAT64_C(   405.00) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -988.34), EASYSIMD_FLOAT64_C(  -370.46), EASYSIMD_FLOAT64_C(  -115.40) },
       INT32_C(          34),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -988.25), EASYSIMD_FLOAT64_C(  -370.25), EASYSIMD_FLOAT64_C(  -115.25) } },
    { { EASYSIMD_FLOAT64_C(  -875.10), EASYSIMD_FLOAT64_C(  -692.94), EASYSIMD_FLOAT64_C(   918.39), EASYSIMD_FLOAT64_C(   -62.11) },
       INT32_C(          50),
      { EASYSIMD_FLOAT64_C(  -875.00), EASYSIMD_FLOAT64_C(  -692.88), EASYSIMD_FLOAT64_C(   918.50), EASYSIMD_FLOAT64_C(   -62.00) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -778.21), EASYSIMD_FLOAT64_C(   646.19), EASYSIMD_FLOAT64_C(   901.45) },
       INT32_C(          66),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -778.19), EASYSIMD_FLOAT64_C(   646.25), EASYSIMD_FLOAT64_C(   901.50) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   -60.39), EASYSIMD_FLOAT64_C(  -944.41), EASYSIMD_FLOAT64_C(   742.26) },
       INT32_C(          82),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   -60.38), EASYSIMD_FLOAT64_C(  -944.41), EASYSIMD_FLOAT64_C(   742.28) } },
    { { EASYSIMD_FLOAT64_C(   730.28),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -594.50), EASYSIMD_FLOAT64_C(  -269.89) },
       INT32_C(          98),
      { EASYSIMD_FLOAT64_C(   730.28),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -594.50), EASYSIMD_FLOAT64_C(  -269.88) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   445.25), EASYSIMD_FLOAT64_C(  -940.50), EASYSIMD_FLOAT64_C(  -511.53) },
       INT32_C(         114),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   445.25), EASYSIMD_FLOAT64_C(  -940.50), EASYSIMD_FLOAT64_C(  -511.52) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -545.25), EASYSIMD_FLOAT64_C(  -781.33), EASYSIMD_FLOAT64_C(  -993.56) },
       INT32_C(         130),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -545.25), EASYSIMD_FLOAT64_C(  -781.33), EASYSIMD_FLOAT64_C(  -993.56) } },
    { { EASYSIMD_FLOAT64_C(   952.68), EASYSIMD_FLOAT64_C(   614.43), EASYSIMD_FLOAT64_C(   203.40), EASYSIMD_FLOAT64_C(   854.13) },
       INT32_C(         146),
      { EASYSIMD_FLOAT64_C(   952.68), EASYSIMD_FLOAT64_C(   614.43), EASYSIMD_FLOAT64_C(   203.40), EASYSIMD_FLOAT64_C(   854.13) } },
    { { EASYSIMD_FLOAT64_C(  -424.40), EASYSIMD_FLOAT64_C(  -640.41), EASYSIMD_FLOAT64_C(  -132.99), EASYSIMD_FLOAT64_C(  -368.81) },
       INT32_C(         162),
      { EASYSIMD_FLOAT64_C(  -424.40), EASYSIMD_FLOAT64_C(  -640.41), EASYSIMD_FLOAT64_C(  -132.99), EASYSIMD_FLOAT64_C(  -368.81) } },
    { { EASYSIMD_FLOAT64_C(  -569.47),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   832.13), EASYSIMD_FLOAT64_C(   841.49) },
       INT32_C(         178),
      { EASYSIMD_FLOAT64_C(  -569.47),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   832.13), EASYSIMD_FLOAT64_C(   841.49) } },
    { { EASYSIMD_FLOAT64_C(   258.07),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   856.76), EASYSIMD_FLOAT64_C(  -296.68) },
       INT32_C(         194),
      { EASYSIMD_FLOAT64_C(   258.07),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   856.76), EASYSIMD_FLOAT64_C(  -296.68) } },
    { { EASYSIMD_FLOAT64_C(  -418.80), EASYSIMD_FLOAT64_C(  -566.71), EASYSIMD_FLOAT64_C(   487.92), EASYSIMD_FLOAT64_C(  -200.13) },
       INT32_C(         210),
      { EASYSIMD_FLOAT64_C(  -418.80), EASYSIMD_FLOAT64_C(  -566.71), EASYSIMD_FLOAT64_C(   487.92), EASYSIMD_FLOAT64_C(  -200.13) } },
    { { EASYSIMD_FLOAT64_C(  -119.44),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -607.58), EASYSIMD_FLOAT64_C(  -505.01) },
       INT32_C(         226),
      { EASYSIMD_FLOAT64_C(  -119.44),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -607.58), EASYSIMD_FLOAT64_C(  -505.01) } },
    { { EASYSIMD_FLOAT64_C(  -863.93), EASYSIMD_FLOAT64_C(  -393.86), EASYSIMD_FLOAT64_C(   289.40), EASYSIMD_FLOAT64_C(  -232.73) },
       INT32_C(         242),
      { EASYSIMD_FLOAT64_C(  -863.93), EASYSIMD_FLOAT64_C(  -393.86), EASYSIMD_FLOAT64_C(   289.40), EASYSIMD_FLOAT64_C(  -232.73) } },
    { { EASYSIMD_FLOAT64_C(   719.93), EASYSIMD_FLOAT64_C(   809.54), EASYSIMD_FLOAT64_C(  -459.89), EASYSIMD_FLOAT64_C(   561.42) },
       INT32_C(           3),
      { EASYSIMD_FLOAT64_C(   719.00), EASYSIMD_FLOAT64_C(   809.00), EASYSIMD_FLOAT64_C(  -459.00), EASYSIMD_FLOAT64_C(   561.00) } },
    { { EASYSIMD_FLOAT64_C(   102.35), EASYSIMD_FLOAT64_C(   559.26), EASYSIMD_FLOAT64_C(  -484.61), EASYSIMD_FLOAT64_C(   712.43) },
       INT32_C(          19),
      { EASYSIMD_FLOAT64_C(   102.00), EASYSIMD_FLOAT64_C(   559.00), EASYSIMD_FLOAT64_C(  -484.50), EASYSIMD_FLOAT64_C(   712.00) } },
    { { EASYSIMD_FLOAT64_C(   218.71),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   761.26), EASYSIMD_FLOAT64_C(  -748.12) },
       INT32_C(          35),
      { EASYSIMD_FLOAT64_C(   218.50),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   761.25), EASYSIMD_FLOAT64_C(  -748.00) } },
    { { EASYSIMD_FLOAT64_C(   763.08),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -379.63), EASYSIMD_FLOAT64_C(  -879.84) },
       INT32_C(          51),
      { EASYSIMD_FLOAT64_C(   763.00),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -379.62), EASYSIMD_FLOAT64_C(  -879.75) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   537.75), EASYSIMD_FLOAT64_C(   816.71), EASYSIMD_FLOAT64_C(   879.38) },
       INT32_C(          67),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   537.75), EASYSIMD_FLOAT64_C(   816.69), EASYSIMD_FLOAT64_C(   879.38) } },
    { { EASYSIMD_FLOAT64_C(  -412.64), EASYSIMD_FLOAT64_C(  -452.92), EASYSIMD_FLOAT64_C(  -606.48), EASYSIMD_FLOAT64_C(   127.47) },
       INT32_C(          83),
      { EASYSIMD_FLOAT64_C(  -412.62), EASYSIMD_FLOAT64_C(  -452.91), EASYSIMD_FLOAT64_C(  -606.47), EASYSIMD_FLOAT64_C(   127.47) } },
    { { EASYSIMD_FLOAT64_C(   650.84),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   667.76), EASYSIMD_FLOAT64_C(  -833.77) },
       INT32_C(          99),
      { EASYSIMD_FLOAT64_C(   650.83),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   667.75), EASYSIMD_FLOAT64_C(  -833.77) } },
    { { EASYSIMD_FLOAT64_C(  -675.74),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   636.82), EASYSIMD_FLOAT64_C(   287.47) },
       INT32_C(         115),
      { EASYSIMD_FLOAT64_C(  -675.73),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   636.81), EASYSIMD_FLOAT64_C(   287.47) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   997.00), EASYSIMD_FLOAT64_C(   170.71), EASYSIMD_FLOAT64_C(  -299.44) },
       INT32_C(         131),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   997.00), EASYSIMD_FLOAT64_C(   170.71), EASYSIMD_FLOAT64_C(  -299.44) } },
    { { EASYSIMD_FLOAT64_C(   973.80),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -331.94), EASYSIMD_FLOAT64_C(   853.18) },
       INT32_C(         147),
      { EASYSIMD_FLOAT64_C(   973.80),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -331.94), EASYSIMD_FLOAT64_C(   853.18) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   645.56), EASYSIMD_FLOAT64_C(   568.01), EASYSIMD_FLOAT64_C(  -867.16) },
       INT32_C(         163),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   645.56), EASYSIMD_FLOAT64_C(   568.01), EASYSIMD_FLOAT64_C(  -867.16) } },
    { { EASYSIMD_FLOAT64_C(   800.61), EASYSIMD_FLOAT64_C(   462.62), EASYSIMD_FLOAT64_C(   740.08), EASYSIMD_FLOAT64_C(  -115.60) },
       INT32_C(         179),
      { EASYSIMD_FLOAT64_C(   800.61), EASYSIMD_FLOAT64_C(   462.62), EASYSIMD_FLOAT64_C(   740.08), EASYSIMD_FLOAT64_C(  -115.60) } },
    { { EASYSIMD_FLOAT64_C(  -935.66),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -515.62), EASYSIMD_FLOAT64_C(   351.80) },
       INT32_C(         195),
      { EASYSIMD_FLOAT64_C(  -935.66),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -515.62), EASYSIMD_FLOAT64_C(   351.80) } },
    { { EASYSIMD_FLOAT64_C(  -557.08), EASYSIMD_FLOAT64_C(   858.00), EASYSIMD_FLOAT64_C(   573.05), EASYSIMD_FLOAT64_C(   143.48) },
       INT32_C(         211),
      { EASYSIMD_FLOAT64_C(  -557.08), EASYSIMD_FLOAT64_C(   858.00), EASYSIMD_FLOAT64_C(   573.05), EASYSIMD_FLOAT64_C(   143.48) } },
    { { EASYSIMD_FLOAT64_C(  -575.60), EASYSIMD_FLOAT64_C(   117.28), EASYSIMD_FLOAT64_C(   620.48), EASYSIMD_FLOAT64_C(    92.46) },
       INT32_C(         227),
      { EASYSIMD_FLOAT64_C(  -575.60), EASYSIMD_FLOAT64_C(   117.28), EASYSIMD_FLOAT64_C(   620.48), EASYSIMD_FLOAT64_C(    92.46) } },
    { { EASYSIMD_FLOAT64_C(    97.75), EASYSIMD_FLOAT64_C(  -655.51), EASYSIMD_FLOAT64_C(   411.01), EASYSIMD_FLOAT64_C(   122.09) },
       INT32_C(         243),
      { EASYSIMD_FLOAT64_C(    97.75), EASYSIMD_FLOAT64_C(  -655.51), EASYSIMD_FLOAT64_C(   411.01), EASYSIMD_FLOAT64_C(   122.09) } },
    { { EASYSIMD_FLOAT64_C(   -20.98), EASYSIMD_FLOAT64_C(   254.94), EASYSIMD_FLOAT64_C(   286.44), EASYSIMD_FLOAT64_C(  -223.15) },
       INT32_C(           4),
      { EASYSIMD_FLOAT64_C(   -21.00), EASYSIMD_FLOAT64_C(   255.00), EASYSIMD_FLOAT64_C(   286.00), EASYSIMD_FLOAT64_C(  -223.00) } },
    { { EASYSIMD_FLOAT64_C(  -250.94), EASYSIMD_FLOAT64_C(  -483.07), EASYSIMD_FLOAT64_C(   939.94), EASYSIMD_FLOAT64_C(   596.62) },
       INT32_C(          20),
      { EASYSIMD_FLOAT64_C(  -251.00), EASYSIMD_FLOAT64_C(  -483.00), EASYSIMD_FLOAT64_C(   940.00), EASYSIMD_FLOAT64_C(   596.50) } },
    { { EASYSIMD_FLOAT64_C(   669.39), EASYSIMD_FLOAT64_C(  -919.01), EASYSIMD_FLOAT64_C(   933.07), EASYSIMD_FLOAT64_C(  -561.56) },
       INT32_C(          36),
      { EASYSIMD_FLOAT64_C(   669.50), EASYSIMD_FLOAT64_C(  -919.00), EASYSIMD_FLOAT64_C(   933.00), EASYSIMD_FLOAT64_C(  -561.50) } },
    { { EASYSIMD_FLOAT64_C(  -664.59), EASYSIMD_FLOAT64_C(  -118.65), EASYSIMD_FLOAT64_C(   800.00), EASYSIMD_FLOAT64_C(   908.47) },
       INT32_C(          52),
      { EASYSIMD_FLOAT64_C(  -664.62), EASYSIMD_FLOAT64_C(  -118.62), EASYSIMD_FLOAT64_C(   800.00), EASYSIMD_FLOAT64_C(   908.50) } },
    { { EASYSIMD_FLOAT64_C(  -229.64), EASYSIMD_FLOAT64_C(  -667.13), EASYSIMD_FLOAT64_C(   142.12), EASYSIMD_FLOAT64_C(  -609.16) },
       INT32_C(          68),
      { EASYSIMD_FLOAT64_C(  -229.62), EASYSIMD_FLOAT64_C(  -667.12), EASYSIMD_FLOAT64_C(   142.12), EASYSIMD_FLOAT64_C(  -609.19) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   488.59), EASYSIMD_FLOAT64_C(   769.82), EASYSIMD_FLOAT64_C(   523.59) },
       INT32_C(          84),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   488.59), EASYSIMD_FLOAT64_C(   769.81), EASYSIMD_FLOAT64_C(   523.59) } },
    { { EASYSIMD_FLOAT64_C(  -497.40), EASYSIMD_FLOAT64_C(   865.61), EASYSIMD_FLOAT64_C(    46.32), EASYSIMD_FLOAT64_C(   279.46) },
       INT32_C(         100),
      { EASYSIMD_FLOAT64_C(  -497.41), EASYSIMD_FLOAT64_C(   865.61), EASYSIMD_FLOAT64_C(    46.31), EASYSIMD_FLOAT64_C(   279.45) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   796.38), EASYSIMD_FLOAT64_C(  -138.90), EASYSIMD_FLOAT64_C(   392.00) },
       INT32_C(         116),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   796.38), EASYSIMD_FLOAT64_C(  -138.90), EASYSIMD_FLOAT64_C(   392.00) } },
    { { EASYSIMD_FLOAT64_C(   472.99),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   -31.06), EASYSIMD_FLOAT64_C(   414.99) },
       INT32_C(         132),
      { EASYSIMD_FLOAT64_C(   472.99),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   -31.06), EASYSIMD_FLOAT64_C(   414.99) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   875.12), EASYSIMD_FLOAT64_C(   985.35), EASYSIMD_FLOAT64_C(  -112.53) },
       INT32_C(         148),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   875.12), EASYSIMD_FLOAT64_C(   985.35), EASYSIMD_FLOAT64_C(  -112.53) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   129.82), EASYSIMD_FLOAT64_C(   864.78), EASYSIMD_FLOAT64_C(  -917.38) },
       INT32_C(         164),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   129.82), EASYSIMD_FLOAT64_C(   864.78), EASYSIMD_FLOAT64_C(  -917.38) } },
    { { EASYSIMD_FLOAT64_C(   842.49),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -658.93), EASYSIMD_FLOAT64_C(  -111.19) },
       INT32_C(         180),
      { EASYSIMD_FLOAT64_C(   842.49),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -658.93), EASYSIMD_FLOAT64_C(  -111.19) } },
    { { EASYSIMD_FLOAT64_C(  -768.16), EASYSIMD_FLOAT64_C(  -876.66), EASYSIMD_FLOAT64_C(  -923.82), EASYSIMD_FLOAT64_C(  -390.51) },
       INT32_C(         196),
      { EASYSIMD_FLOAT64_C(  -768.16), EASYSIMD_FLOAT64_C(  -876.66), EASYSIMD_FLOAT64_C(  -923.82), EASYSIMD_FLOAT64_C(  -390.51) } },
    { { EASYSIMD_FLOAT64_C(   549.18), EASYSIMD_FLOAT64_C(   -79.79), EASYSIMD_FLOAT64_C(   622.77), EASYSIMD_FLOAT64_C(   -35.84) },
       INT32_C(         212),
      { EASYSIMD_FLOAT64_C(   549.18), EASYSIMD_FLOAT64_C(   -79.79), EASYSIMD_FLOAT64_C(   622.77), EASYSIMD_FLOAT64_C(   -35.84) } },
    { { EASYSIMD_FLOAT64_C(   473.06), EASYSIMD_FLOAT64_C(  -820.85), EASYSIMD_FLOAT64_C(  -879.06), EASYSIMD_FLOAT64_C(   348.18) },
       INT32_C(         228),
      { EASYSIMD_FLOAT64_C(   473.06), EASYSIMD_FLOAT64_C(  -820.85), EASYSIMD_FLOAT64_C(  -879.06), EASYSIMD_FLOAT64_C(   348.18) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -634.59), EASYSIMD_FLOAT64_C(  -459.31), EASYSIMD_FLOAT64_C(   321.20) },
       INT32_C(         244),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -634.59), EASYSIMD_FLOAT64_C(  -459.31), EASYSIMD_FLOAT64_C(   321.20) } },
  };

  easysimd__m256d a, r;

  a = easysimd_mm256_loadu_pd(test_vec[0].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(           0));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[1].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          16));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[2].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          32));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[3].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          48));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[4].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          64));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[5].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          80));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[6].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          96));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[7].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         112));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[7].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[8].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         128));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[8].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[9].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         144));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[9].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[10].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         160));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[10].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[11].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         176));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[11].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[12].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         192));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[12].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[13].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         208));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[13].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[14].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         224));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[14].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[15].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         240));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[15].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[16].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(           1));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[16].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[17].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          17));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[17].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[18].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          33));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[18].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[19].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          49));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[19].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[20].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          65));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[20].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[21].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          81));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[21].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[22].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          97));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[22].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[23].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         113));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[23].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[24].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         129));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[24].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[25].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         145));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[25].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[26].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         161));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[26].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[27].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         177));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[27].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[28].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         193));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[28].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[29].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         209));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[29].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[30].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         225));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[30].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[31].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         241));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[31].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[32].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(           2));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[32].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[33].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          18));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[33].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[34].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          34));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[34].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[35].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          50));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[35].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[36].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          66));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[36].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[37].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          82));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[37].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[38].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          98));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[38].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[39].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         114));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[39].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[40].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         130));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[40].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[41].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         146));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[41].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[42].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         162));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[42].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[43].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         178));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[43].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[44].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         194));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[44].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[45].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         210));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[45].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[46].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         226));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[46].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[47].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         242));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[47].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[48].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(           3));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[48].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[49].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          19));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[49].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[50].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          35));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[50].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[51].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          51));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[51].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[52].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          67));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[52].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[53].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          83));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[53].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[54].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          99));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[54].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[55].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         115));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[55].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[56].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         131));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[56].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[57].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         147));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[57].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[58].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         163));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[58].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[59].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         179));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[59].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[60].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         195));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[60].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[61].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         211));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[61].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[62].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         227));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[62].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[63].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         243));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[63].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[64].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(           4));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[64].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[65].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          20));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[65].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[66].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          36));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[66].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[67].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          52));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[67].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[68].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          68));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[68].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[69].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(          84));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[69].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[70].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         100));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[70].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[71].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         116));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[71].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[72].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         132));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[72].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[73].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         148));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[73].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[74].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         164));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[74].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[75].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         180));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[75].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[76].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         196));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[76].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[77].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         212));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[77].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[78].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         228));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[78].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[79].a);
  r = easysimd_mm256_roundscale_pd(a, INT32_C(         244));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[79].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 5 ; i++) {
    for (int j = 0 ; j < 16 ; j++) {
      easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
      if((easysimd_test_codegen_rand() & 1)) {
        if((easysimd_test_codegen_rand() & 1))
          a = easysimd_mm256_blend_pd(a, easysimd_mm256_set1_pd(EASYSIMD_MATH_NAN), 1);
        else {
          if((easysimd_test_codegen_rand() & 1))
            a = easysimd_mm256_blend_pd(a, easysimd_mm256_set1_pd(EASYSIMD_MATH_INFINITY), 2);
          else
            a = easysimd_mm256_blend_pd(a, easysimd_mm256_set1_pd(-EASYSIMD_MATH_INFINITY), 2);
        }
      }
      int imm8 = ((j << 4) | round_type[i]) & 255;
      easysimd__m256d r;
      EASYSIMD_CONSTIFY_256_(easysimd_mm256_roundscale_pd, r, easysimd_mm256_setzero_pd(), imm8, a);

      easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
      easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
      easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
    }
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_roundscale_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[4];
    const easysimd__mmask8 k;
    const easysimd_float64 a[4];
    const int32_t imm8;
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   926.29), EASYSIMD_FLOAT64_C(   909.09), EASYSIMD_FLOAT64_C(   774.77), EASYSIMD_FLOAT64_C(   372.04) },
      UINT8_C(125),
      { EASYSIMD_FLOAT64_C(   458.74), EASYSIMD_FLOAT64_C(  -735.76), EASYSIMD_FLOAT64_C(   634.56), EASYSIMD_FLOAT64_C(   216.61) },
       INT32_C(          32),
      { EASYSIMD_FLOAT64_C(   458.75), EASYSIMD_FLOAT64_C(   909.09), EASYSIMD_FLOAT64_C(   634.50), EASYSIMD_FLOAT64_C(   216.50) } },
    { { EASYSIMD_FLOAT64_C(   808.28), EASYSIMD_FLOAT64_C(  -973.92), EASYSIMD_FLOAT64_C(   364.29), EASYSIMD_FLOAT64_C(   260.83) },
      UINT8_C( 21),
      { EASYSIMD_FLOAT64_C(   685.29), EASYSIMD_FLOAT64_C(  -217.73), EASYSIMD_FLOAT64_C(   979.97), EASYSIMD_FLOAT64_C(   463.75) },
       INT32_C(          81),
      { EASYSIMD_FLOAT64_C(   685.28), EASYSIMD_FLOAT64_C(  -973.92), EASYSIMD_FLOAT64_C(   979.97), EASYSIMD_FLOAT64_C(   260.83) } },
    { { EASYSIMD_FLOAT64_C(   526.14), EASYSIMD_FLOAT64_C(   931.62), EASYSIMD_FLOAT64_C(   993.03), EASYSIMD_FLOAT64_C(   390.55) },
      UINT8_C( 90),
      { EASYSIMD_FLOAT64_C(  -182.65),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   177.32), EASYSIMD_FLOAT64_C(  -823.86) },
       INT32_C(          18),
      { EASYSIMD_FLOAT64_C(   526.14),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   993.03), EASYSIMD_FLOAT64_C(  -823.50) } },
    { { EASYSIMD_FLOAT64_C(    34.50), EASYSIMD_FLOAT64_C(   409.65), EASYSIMD_FLOAT64_C(  -471.60), EASYSIMD_FLOAT64_C(  -330.93) },
      UINT8_C(230),
      { EASYSIMD_FLOAT64_C(  -757.14), EASYSIMD_FLOAT64_C(  -817.96), EASYSIMD_FLOAT64_C(  -565.47), EASYSIMD_FLOAT64_C(  -731.06) },
       INT32_C(         163),
      { EASYSIMD_FLOAT64_C(    34.50), EASYSIMD_FLOAT64_C(  -817.96), EASYSIMD_FLOAT64_C(  -565.47), EASYSIMD_FLOAT64_C(  -330.93) } },
    { { EASYSIMD_FLOAT64_C(  -509.07), EASYSIMD_FLOAT64_C(   231.61), EASYSIMD_FLOAT64_C(  -522.37), EASYSIMD_FLOAT64_C(  -529.09) },
      UINT8_C( 55),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   -49.42), EASYSIMD_FLOAT64_C(  -778.50), EASYSIMD_FLOAT64_C(   394.59) },
       INT32_C(          84),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   -49.41), EASYSIMD_FLOAT64_C(  -778.50), EASYSIMD_FLOAT64_C(  -529.09) } },
    { { EASYSIMD_FLOAT64_C(   760.96), EASYSIMD_FLOAT64_C(  -422.11), EASYSIMD_FLOAT64_C(  -513.55), EASYSIMD_FLOAT64_C(   937.10) },
      UINT8_C( 63),
      { EASYSIMD_FLOAT64_C(   572.86), EASYSIMD_FLOAT64_C(   888.00), EASYSIMD_FLOAT64_C(   734.18), EASYSIMD_FLOAT64_C(  -392.63) },
       INT32_C(          80),
      { EASYSIMD_FLOAT64_C(   572.88), EASYSIMD_FLOAT64_C(   888.00), EASYSIMD_FLOAT64_C(   734.19), EASYSIMD_FLOAT64_C(  -392.62) } },
    { { EASYSIMD_FLOAT64_C(   276.43), EASYSIMD_FLOAT64_C(   923.91), EASYSIMD_FLOAT64_C(  -494.56), EASYSIMD_FLOAT64_C(   458.47) },
      UINT8_C(252),
      { EASYSIMD_FLOAT64_C(  -225.62),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -946.20), EASYSIMD_FLOAT64_C(   265.31) },
       INT32_C(          65),
      { EASYSIMD_FLOAT64_C(   276.43), EASYSIMD_FLOAT64_C(   923.91), EASYSIMD_FLOAT64_C(  -946.25), EASYSIMD_FLOAT64_C(   265.25) } },
    { { EASYSIMD_FLOAT64_C(   994.40), EASYSIMD_FLOAT64_C(  -313.20), EASYSIMD_FLOAT64_C(   153.28), EASYSIMD_FLOAT64_C(   388.99) },
      UINT8_C(122),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -301.89), EASYSIMD_FLOAT64_C(   391.38), EASYSIMD_FLOAT64_C(   343.23) },
       INT32_C(         178),
      { EASYSIMD_FLOAT64_C(   994.40), EASYSIMD_FLOAT64_C(  -301.89), EASYSIMD_FLOAT64_C(   153.28), EASYSIMD_FLOAT64_C(   343.23) } },
  };

  easysimd__m256d src, a, r;

  src = easysimd_mm256_loadu_pd(test_vec[0].src);
  a = easysimd_mm256_loadu_pd(test_vec[0].a);
  r = easysimd_mm256_mask_roundscale_pd(src, test_vec[0].k, a, INT32_C(          32));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[0].r), 1);

  src = easysimd_mm256_loadu_pd(test_vec[1].src);
  a = easysimd_mm256_loadu_pd(test_vec[1].a);
  r = easysimd_mm256_mask_roundscale_pd(src, test_vec[1].k, a, INT32_C(          81));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[1].r), 1);

  src = easysimd_mm256_loadu_pd(test_vec[2].src);
  a = easysimd_mm256_loadu_pd(test_vec[2].a);
  r = easysimd_mm256_mask_roundscale_pd(src, test_vec[2].k, a, INT32_C(          18));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[2].r), 1);

  src = easysimd_mm256_loadu_pd(test_vec[3].src);
  a = easysimd_mm256_loadu_pd(test_vec[3].a);
  r = easysimd_mm256_mask_roundscale_pd(src, test_vec[3].k, a, INT32_C(         163));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[3].r), 1);

  src = easysimd_mm256_loadu_pd(test_vec[4].src);
  a = easysimd_mm256_loadu_pd(test_vec[4].a);
  r = easysimd_mm256_mask_roundscale_pd(src, test_vec[4].k, a, INT32_C(          84));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[4].r), 1);

  src = easysimd_mm256_loadu_pd(test_vec[5].src);
  a = easysimd_mm256_loadu_pd(test_vec[5].a);
  r = easysimd_mm256_mask_roundscale_pd(src, test_vec[5].k, a, INT32_C(          80));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[5].r), 1);

  src = easysimd_mm256_loadu_pd(test_vec[6].src);
  a = easysimd_mm256_loadu_pd(test_vec[6].a);
  r = easysimd_mm256_mask_roundscale_pd(src, test_vec[6].k, a, INT32_C(          65));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[6].r), 1);

  src = easysimd_mm256_loadu_pd(test_vec[7].src);
  a = easysimd_mm256_loadu_pd(test_vec[7].a);
  r = easysimd_mm256_mask_roundscale_pd(src, test_vec[7].k, a, INT32_C(         178));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256d src = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    if((easysimd_test_codegen_rand() & 1)) {
      if((easysimd_test_codegen_rand() & 1))
        a = easysimd_mm256_blend_pd(a, easysimd_mm256_set1_pd(EASYSIMD_MATH_NAN), 1);
      else {
        if((easysimd_test_codegen_rand() & 1))
          a = easysimd_mm256_blend_pd(a, easysimd_mm256_set1_pd(EASYSIMD_MATH_INFINITY), 2);
        else
          a = easysimd_mm256_blend_pd(a, easysimd_mm256_set1_pd(-EASYSIMD_MATH_INFINITY), 2);
      }
    }
    int imm8 = (((easysimd_test_codegen_rand() & 15) << 4) | round_type[i % 5]) & 255;
    easysimd__m256d r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm256_mask_roundscale_pd, r, easysimd_mm256_setzero_pd(), imm8, src, k, a);

    easysimd_test_x86_write_f64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_roundscale_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[4];
    const int32_t imm8;
    const easysimd_float64 r[4];
  } test_vec[] = {
    { UINT8_C(131),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -452.58), EASYSIMD_FLOAT64_C(   364.79), EASYSIMD_FLOAT64_C(  -485.87) },
       INT32_C(         144),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -452.58), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(237),
      { EASYSIMD_FLOAT64_C(  -900.30), EASYSIMD_FLOAT64_C(  -203.53), EASYSIMD_FLOAT64_C(  -910.18), EASYSIMD_FLOAT64_C(   104.50) },
       INT32_C(         161),
      { EASYSIMD_FLOAT64_C(  -900.30), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -910.18), EASYSIMD_FLOAT64_C(   104.50) } },
    { UINT8_C(139),
      { EASYSIMD_FLOAT64_C(   381.69), EASYSIMD_FLOAT64_C(    91.35), EASYSIMD_FLOAT64_C(  -727.30), EASYSIMD_FLOAT64_C(   376.09) },
       INT32_C(         146),
      { EASYSIMD_FLOAT64_C(   381.69), EASYSIMD_FLOAT64_C(    91.35), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   376.09) } },
    { UINT8_C(120),
      { EASYSIMD_FLOAT64_C(   408.57), EASYSIMD_FLOAT64_C(  -808.69), EASYSIMD_FLOAT64_C(   463.20), EASYSIMD_FLOAT64_C(  -200.06) },
       INT32_C(         211),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -200.06) } },
    { UINT8_C(226),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   405.18), EASYSIMD_FLOAT64_C(   344.89), EASYSIMD_FLOAT64_C(  -104.81) },
       INT32_C(          68),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   405.19), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(234),
      { EASYSIMD_FLOAT64_C(  -702.95), EASYSIMD_FLOAT64_C(    20.62), EASYSIMD_FLOAT64_C(   510.88), EASYSIMD_FLOAT64_C(    93.52) },
       INT32_C(          32),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    20.50), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    93.50) } },
    { UINT8_C(113),
      { EASYSIMD_FLOAT64_C(  -534.43), EASYSIMD_FLOAT64_C(   956.30), EASYSIMD_FLOAT64_C(   325.48), EASYSIMD_FLOAT64_C(   556.92) },
       INT32_C(         209),
      { EASYSIMD_FLOAT64_C(  -534.43), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(169),
      { EASYSIMD_FLOAT64_C(   654.97), EASYSIMD_FLOAT64_C(   466.66), EASYSIMD_FLOAT64_C(  -256.36), EASYSIMD_FLOAT64_C(   846.28) },
       INT32_C(          50),
      { EASYSIMD_FLOAT64_C(   655.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   846.38) } },
  };

  easysimd__m256d a, r;

  a = easysimd_mm256_loadu_pd(test_vec[0].a);
  r = easysimd_mm256_maskz_roundscale_pd(test_vec[0].k, a, INT32_C(         144));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[1].a);
  r = easysimd_mm256_maskz_roundscale_pd(test_vec[1].k, a, INT32_C(         161));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[2].a);
  r = easysimd_mm256_maskz_roundscale_pd(test_vec[2].k, a, INT32_C(         146));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[3].a);
  r = easysimd_mm256_maskz_roundscale_pd(test_vec[3].k, a, INT32_C(         211));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[4].a);
  r = easysimd_mm256_maskz_roundscale_pd(test_vec[4].k, a, INT32_C(          68));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[5].a);
  r = easysimd_mm256_maskz_roundscale_pd(test_vec[5].k, a, INT32_C(          32));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[6].a);
  r = easysimd_mm256_maskz_roundscale_pd(test_vec[6].k, a, INT32_C(         209));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[7].a);
  r = easysimd_mm256_maskz_roundscale_pd(test_vec[7].k, a, INT32_C(          50));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    if((easysimd_test_codegen_rand() & 1)) {
      if((easysimd_test_codegen_rand() & 1))
        a = easysimd_mm256_blend_pd(a, easysimd_mm256_set1_pd(EASYSIMD_MATH_NAN), 1);
      else {
        if((easysimd_test_codegen_rand() & 1))
          a = easysimd_mm256_blend_pd(a, easysimd_mm256_set1_pd(EASYSIMD_MATH_INFINITY), 2);
        else
          a = easysimd_mm256_blend_pd(a, easysimd_mm256_set1_pd(-EASYSIMD_MATH_INFINITY), 2);
      }
    }
    int imm8 = (((easysimd_test_codegen_rand() & 15) << 4) | round_type[i % 5]) & 255;
    easysimd__m256d r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm256_maskz_roundscale_pd, r, easysimd_mm256_setzero_pd(), imm8, k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_roundscale_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[8];
    const int32_t imm8;
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   -67.30),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -705.81), EASYSIMD_FLOAT64_C(   196.47),
        EASYSIMD_FLOAT64_C(   177.22), EASYSIMD_FLOAT64_C(   391.26), EASYSIMD_FLOAT64_C(   -54.56), EASYSIMD_FLOAT64_C(   829.93) },
       INT32_C(         224),
      { EASYSIMD_FLOAT64_C(   -67.30),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -705.81), EASYSIMD_FLOAT64_C(   196.47),
        EASYSIMD_FLOAT64_C(   177.22), EASYSIMD_FLOAT64_C(   391.26), EASYSIMD_FLOAT64_C(   -54.56), EASYSIMD_FLOAT64_C(   829.93) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   889.83), EASYSIMD_FLOAT64_C(   441.33), EASYSIMD_FLOAT64_C(  -977.58),
        EASYSIMD_FLOAT64_C(  -699.06), EASYSIMD_FLOAT64_C(   260.67), EASYSIMD_FLOAT64_C(  -418.77), EASYSIMD_FLOAT64_C(   157.04) },
       INT32_C(         129),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   889.83), EASYSIMD_FLOAT64_C(   441.33), EASYSIMD_FLOAT64_C(  -977.58),
        EASYSIMD_FLOAT64_C(  -699.06), EASYSIMD_FLOAT64_C(   260.67), EASYSIMD_FLOAT64_C(  -418.77), EASYSIMD_FLOAT64_C(   157.04) } },
    { { EASYSIMD_FLOAT64_C(   -38.35),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -257.79), EASYSIMD_FLOAT64_C(  -975.63),
        EASYSIMD_FLOAT64_C(   411.77), EASYSIMD_FLOAT64_C(   590.64), EASYSIMD_FLOAT64_C(   -80.22), EASYSIMD_FLOAT64_C(   714.31) },
       INT32_C(          82),
      { EASYSIMD_FLOAT64_C(   -38.34),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -257.78), EASYSIMD_FLOAT64_C(  -975.62),
        EASYSIMD_FLOAT64_C(   411.78), EASYSIMD_FLOAT64_C(   590.66), EASYSIMD_FLOAT64_C(   -80.22), EASYSIMD_FLOAT64_C(   714.31) } },
    { { EASYSIMD_FLOAT64_C(  -903.12), EASYSIMD_FLOAT64_C(   399.77), EASYSIMD_FLOAT64_C(  -334.76), EASYSIMD_FLOAT64_C(   926.81),
        EASYSIMD_FLOAT64_C(  -852.06), EASYSIMD_FLOAT64_C(  -961.10), EASYSIMD_FLOAT64_C(   107.72), EASYSIMD_FLOAT64_C(   666.94) },
       INT32_C(          51),
      { EASYSIMD_FLOAT64_C(  -903.00), EASYSIMD_FLOAT64_C(   399.75), EASYSIMD_FLOAT64_C(  -334.75), EASYSIMD_FLOAT64_C(   926.75),
        EASYSIMD_FLOAT64_C(  -852.00), EASYSIMD_FLOAT64_C(  -961.00), EASYSIMD_FLOAT64_C(   107.62), EASYSIMD_FLOAT64_C(   666.88) } },
    { { EASYSIMD_FLOAT64_C(   108.27), EASYSIMD_FLOAT64_C(  -600.58), EASYSIMD_FLOAT64_C(   298.49), EASYSIMD_FLOAT64_C(  -631.06),
        EASYSIMD_FLOAT64_C(   -19.34), EASYSIMD_FLOAT64_C(  -544.47), EASYSIMD_FLOAT64_C(  -798.15), EASYSIMD_FLOAT64_C(   358.47) },
       INT32_C(          68),
      { EASYSIMD_FLOAT64_C(   108.25), EASYSIMD_FLOAT64_C(  -600.56), EASYSIMD_FLOAT64_C(   298.50), EASYSIMD_FLOAT64_C(  -631.06),
        EASYSIMD_FLOAT64_C(   -19.31), EASYSIMD_FLOAT64_C(  -544.50), EASYSIMD_FLOAT64_C(  -798.12), EASYSIMD_FLOAT64_C(   358.50) } },
    { { EASYSIMD_FLOAT64_C(  -893.30), EASYSIMD_FLOAT64_C(    50.96), EASYSIMD_FLOAT64_C(   187.87), EASYSIMD_FLOAT64_C(   518.48),
        EASYSIMD_FLOAT64_C(  -358.41), EASYSIMD_FLOAT64_C(  -892.35), EASYSIMD_FLOAT64_C(   232.79), EASYSIMD_FLOAT64_C(   164.93) },
       INT32_C(         224),
      { EASYSIMD_FLOAT64_C(  -893.30), EASYSIMD_FLOAT64_C(    50.96), EASYSIMD_FLOAT64_C(   187.87), EASYSIMD_FLOAT64_C(   518.48),
        EASYSIMD_FLOAT64_C(  -358.41), EASYSIMD_FLOAT64_C(  -892.35), EASYSIMD_FLOAT64_C(   232.79), EASYSIMD_FLOAT64_C(   164.93) } },
    { { EASYSIMD_FLOAT64_C(  -115.27), EASYSIMD_FLOAT64_C(   124.20), EASYSIMD_FLOAT64_C(  -358.93), EASYSIMD_FLOAT64_C(   549.98),
        EASYSIMD_FLOAT64_C(    51.01), EASYSIMD_FLOAT64_C(  -211.00), EASYSIMD_FLOAT64_C(   588.88), EASYSIMD_FLOAT64_C(  -841.27) },
       INT32_C(         241),
      { EASYSIMD_FLOAT64_C(  -115.27), EASYSIMD_FLOAT64_C(   124.20), EASYSIMD_FLOAT64_C(  -358.93), EASYSIMD_FLOAT64_C(   549.98),
        EASYSIMD_FLOAT64_C(    51.01), EASYSIMD_FLOAT64_C(  -211.00), EASYSIMD_FLOAT64_C(   588.88), EASYSIMD_FLOAT64_C(  -841.27) } },
    { { EASYSIMD_FLOAT64_C(   156.28), EASYSIMD_FLOAT64_C(   564.22), EASYSIMD_FLOAT64_C(  -634.69), EASYSIMD_FLOAT64_C(  -545.23),
        EASYSIMD_FLOAT64_C(   933.16), EASYSIMD_FLOAT64_C(   345.96), EASYSIMD_FLOAT64_C(   -89.70), EASYSIMD_FLOAT64_C(  -864.99) },
       INT32_C(         162),
      { EASYSIMD_FLOAT64_C(   156.28), EASYSIMD_FLOAT64_C(   564.22), EASYSIMD_FLOAT64_C(  -634.69), EASYSIMD_FLOAT64_C(  -545.23),
        EASYSIMD_FLOAT64_C(   933.16), EASYSIMD_FLOAT64_C(   345.96), EASYSIMD_FLOAT64_C(   -89.70), EASYSIMD_FLOAT64_C(  -864.99) } },
  };

  easysimd__m512d a, r;

  a = easysimd_mm512_loadu_pd(test_vec[0].a);
  r = easysimd_mm512_roundscale_pd(a, INT32_C(         224));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[1].a);
  r = easysimd_mm512_roundscale_pd(a, INT32_C(         129));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[2].a);
  r = easysimd_mm512_roundscale_pd(a, INT32_C(          82));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[3].a);
  r = easysimd_mm512_roundscale_pd(a, INT32_C(          51));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[4].a);
  r = easysimd_mm512_roundscale_pd(a, INT32_C(          68));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[5].a);
  r = easysimd_mm512_roundscale_pd(a, INT32_C(         224));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[6].a);
  r = easysimd_mm512_roundscale_pd(a, INT32_C(         241));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[7].a);
  r = easysimd_mm512_roundscale_pd(a, INT32_C(         162));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    if((easysimd_test_codegen_rand() & 1)) {
      if((easysimd_test_codegen_rand() & 1))
        a = easysimd_mm512_mask_mov_pd(a, 1, easysimd_mm512_set1_pd(EASYSIMD_MATH_NAN));
      else {
        if((easysimd_test_codegen_rand() & 1))
          a = easysimd_mm512_mask_mov_pd(a, 2, easysimd_mm512_set1_pd(EASYSIMD_MATH_INFINITY));
        else
          a = easysimd_mm512_mask_mov_pd(a, 2, easysimd_mm512_set1_pd(-EASYSIMD_MATH_INFINITY));
      }
    }
    int imm8 = (((easysimd_test_codegen_rand() & 15) << 4) | round_type[i % 5]) & 255;
    easysimd__m512d r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm512_roundscale_pd, r, easysimd_mm512_setzero_pd(), imm8, a);

    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_roundscale_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[8];
    const easysimd__mmask8 k;
    const easysimd_float64 a[8];
    const int32_t imm8;
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -157.27), EASYSIMD_FLOAT64_C(   107.68), EASYSIMD_FLOAT64_C(   630.83), EASYSIMD_FLOAT64_C(  -905.04),
        EASYSIMD_FLOAT64_C(  -496.50), EASYSIMD_FLOAT64_C(   850.05), EASYSIMD_FLOAT64_C(  -847.15), EASYSIMD_FLOAT64_C(  -488.99) },
      UINT8_C( 82),
      { EASYSIMD_FLOAT64_C(    92.08),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   673.37), EASYSIMD_FLOAT64_C(  -818.01),
        EASYSIMD_FLOAT64_C(  -755.51), EASYSIMD_FLOAT64_C(  -570.18), EASYSIMD_FLOAT64_C(  -127.51), EASYSIMD_FLOAT64_C(    50.84) },
       INT32_C(         192),
      { EASYSIMD_FLOAT64_C(  -157.27),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   630.83), EASYSIMD_FLOAT64_C(  -905.04),
        EASYSIMD_FLOAT64_C(  -755.51), EASYSIMD_FLOAT64_C(   850.05), EASYSIMD_FLOAT64_C(  -127.51), EASYSIMD_FLOAT64_C(  -488.99) } },
    { { EASYSIMD_FLOAT64_C(  -447.85), EASYSIMD_FLOAT64_C(  -701.14), EASYSIMD_FLOAT64_C(   316.16), EASYSIMD_FLOAT64_C(   117.46),
        EASYSIMD_FLOAT64_C(   360.05), EASYSIMD_FLOAT64_C(   -16.49), EASYSIMD_FLOAT64_C(   900.84), EASYSIMD_FLOAT64_C(  -215.65) },
      UINT8_C(133),
      { EASYSIMD_FLOAT64_C(   239.19), EASYSIMD_FLOAT64_C(   627.08), EASYSIMD_FLOAT64_C(   817.84), EASYSIMD_FLOAT64_C(  -129.99),
        EASYSIMD_FLOAT64_C(   722.04), EASYSIMD_FLOAT64_C(  -678.66), EASYSIMD_FLOAT64_C(  -279.94), EASYSIMD_FLOAT64_C(   874.89) },
       INT32_C(          81),
      { EASYSIMD_FLOAT64_C(   239.19), EASYSIMD_FLOAT64_C(  -701.14), EASYSIMD_FLOAT64_C(   817.81), EASYSIMD_FLOAT64_C(   117.46),
        EASYSIMD_FLOAT64_C(   360.05), EASYSIMD_FLOAT64_C(   -16.49), EASYSIMD_FLOAT64_C(   900.84), EASYSIMD_FLOAT64_C(   874.88) } },
    { { EASYSIMD_FLOAT64_C(   -33.03), EASYSIMD_FLOAT64_C(  -990.49), EASYSIMD_FLOAT64_C(  -243.90), EASYSIMD_FLOAT64_C(   148.97),
        EASYSIMD_FLOAT64_C(  -746.00), EASYSIMD_FLOAT64_C(   185.92), EASYSIMD_FLOAT64_C(  -978.54), EASYSIMD_FLOAT64_C(   304.84) },
      UINT8_C(135),
      { EASYSIMD_FLOAT64_C(   255.00), EASYSIMD_FLOAT64_C(   685.58), EASYSIMD_FLOAT64_C(  -123.66), EASYSIMD_FLOAT64_C(   807.15),
        EASYSIMD_FLOAT64_C(   984.44), EASYSIMD_FLOAT64_C(  -807.49), EASYSIMD_FLOAT64_C(   -75.38), EASYSIMD_FLOAT64_C(   344.49) },
       INT32_C(         242),
      { EASYSIMD_FLOAT64_C(   255.00), EASYSIMD_FLOAT64_C(   685.58), EASYSIMD_FLOAT64_C(  -123.66), EASYSIMD_FLOAT64_C(   148.97),
        EASYSIMD_FLOAT64_C(  -746.00), EASYSIMD_FLOAT64_C(   185.92), EASYSIMD_FLOAT64_C(  -978.54), EASYSIMD_FLOAT64_C(   344.49) } },
    { { EASYSIMD_FLOAT64_C(  -871.16), EASYSIMD_FLOAT64_C(   886.18), EASYSIMD_FLOAT64_C(  -935.35), EASYSIMD_FLOAT64_C(   755.92),
        EASYSIMD_FLOAT64_C(   704.02), EASYSIMD_FLOAT64_C(   -65.34), EASYSIMD_FLOAT64_C(   477.96), EASYSIMD_FLOAT64_C(  -974.64) },
      UINT8_C( 57),
      { EASYSIMD_FLOAT64_C(   352.85), EASYSIMD_FLOAT64_C(  -142.29), EASYSIMD_FLOAT64_C(  -262.55), EASYSIMD_FLOAT64_C(  -680.18),
        EASYSIMD_FLOAT64_C(  -132.78), EASYSIMD_FLOAT64_C(   493.55), EASYSIMD_FLOAT64_C(   468.79), EASYSIMD_FLOAT64_C(   121.22) },
       INT32_C(          99),
      { EASYSIMD_FLOAT64_C(   352.84), EASYSIMD_FLOAT64_C(   886.18), EASYSIMD_FLOAT64_C(  -935.35), EASYSIMD_FLOAT64_C(  -680.17),
        EASYSIMD_FLOAT64_C(  -132.77), EASYSIMD_FLOAT64_C(   493.55), EASYSIMD_FLOAT64_C(   477.96), EASYSIMD_FLOAT64_C(  -974.64) } },
    { { EASYSIMD_FLOAT64_C(  -573.94), EASYSIMD_FLOAT64_C(   992.80), EASYSIMD_FLOAT64_C(  -254.75), EASYSIMD_FLOAT64_C(  -888.36),
        EASYSIMD_FLOAT64_C(  -130.86), EASYSIMD_FLOAT64_C(  -447.59), EASYSIMD_FLOAT64_C(  -903.92), EASYSIMD_FLOAT64_C(    61.65) },
      UINT8_C(208),
      { EASYSIMD_FLOAT64_C(   440.58), EASYSIMD_FLOAT64_C(  -762.33), EASYSIMD_FLOAT64_C(  -697.52), EASYSIMD_FLOAT64_C(   569.41),
        EASYSIMD_FLOAT64_C(  -876.15), EASYSIMD_FLOAT64_C(  -632.88), EASYSIMD_FLOAT64_C(   325.33), EASYSIMD_FLOAT64_C(   827.87) },
       INT32_C(           4),
      { EASYSIMD_FLOAT64_C(  -573.94), EASYSIMD_FLOAT64_C(   992.80), EASYSIMD_FLOAT64_C(  -254.75), EASYSIMD_FLOAT64_C(  -888.36),
        EASYSIMD_FLOAT64_C(  -876.00), EASYSIMD_FLOAT64_C(  -447.59), EASYSIMD_FLOAT64_C(   325.00), EASYSIMD_FLOAT64_C(   828.00) } },
    { { EASYSIMD_FLOAT64_C(   853.23), EASYSIMD_FLOAT64_C(   -43.49), EASYSIMD_FLOAT64_C(  -843.86), EASYSIMD_FLOAT64_C(  -289.06),
        EASYSIMD_FLOAT64_C(   693.96), EASYSIMD_FLOAT64_C(  -524.04), EASYSIMD_FLOAT64_C(   578.16), EASYSIMD_FLOAT64_C(   187.50) },
      UINT8_C( 25),
      { EASYSIMD_FLOAT64_C(  -300.62),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   435.01), EASYSIMD_FLOAT64_C(   125.44),
        EASYSIMD_FLOAT64_C(   859.76), EASYSIMD_FLOAT64_C(  -819.74), EASYSIMD_FLOAT64_C(   237.08), EASYSIMD_FLOAT64_C(  -271.10) },
       INT32_C(         192),
      { EASYSIMD_FLOAT64_C(  -300.62), EASYSIMD_FLOAT64_C(   -43.49), EASYSIMD_FLOAT64_C(  -843.86), EASYSIMD_FLOAT64_C(   125.44),
        EASYSIMD_FLOAT64_C(   859.76), EASYSIMD_FLOAT64_C(  -524.04), EASYSIMD_FLOAT64_C(   578.16), EASYSIMD_FLOAT64_C(   187.50) } },
    { { EASYSIMD_FLOAT64_C(  -226.26), EASYSIMD_FLOAT64_C(  -971.78), EASYSIMD_FLOAT64_C(  -487.83), EASYSIMD_FLOAT64_C(  -656.85),
        EASYSIMD_FLOAT64_C(  -847.93), EASYSIMD_FLOAT64_C(  -120.71), EASYSIMD_FLOAT64_C(   668.48), EASYSIMD_FLOAT64_C(   979.94) },
      UINT8_C( 89),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   833.17), EASYSIMD_FLOAT64_C(   137.58), EASYSIMD_FLOAT64_C(  -372.09),
        EASYSIMD_FLOAT64_C(  -455.89), EASYSIMD_FLOAT64_C(  -168.46), EASYSIMD_FLOAT64_C(   103.87), EASYSIMD_FLOAT64_C(  -877.73) },
       INT32_C(         177),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -971.78), EASYSIMD_FLOAT64_C(  -487.83), EASYSIMD_FLOAT64_C(  -372.09),
        EASYSIMD_FLOAT64_C(  -455.89), EASYSIMD_FLOAT64_C(  -120.71), EASYSIMD_FLOAT64_C(   103.87), EASYSIMD_FLOAT64_C(   979.94) } },
    { { EASYSIMD_FLOAT64_C(   886.00), EASYSIMD_FLOAT64_C(  -516.36), EASYSIMD_FLOAT64_C(   947.10), EASYSIMD_FLOAT64_C(   745.77),
        EASYSIMD_FLOAT64_C(  -336.10), EASYSIMD_FLOAT64_C(   184.18), EASYSIMD_FLOAT64_C(  -525.33), EASYSIMD_FLOAT64_C(   396.56) },
      UINT8_C(254),
      { EASYSIMD_FLOAT64_C(  -734.78), EASYSIMD_FLOAT64_C(   606.25), EASYSIMD_FLOAT64_C(   291.08), EASYSIMD_FLOAT64_C(  -706.55),
        EASYSIMD_FLOAT64_C(  -881.59), EASYSIMD_FLOAT64_C(   634.23), EASYSIMD_FLOAT64_C(  -554.48), EASYSIMD_FLOAT64_C(    -2.30) },
       INT32_C(         178),
      { EASYSIMD_FLOAT64_C(   886.00), EASYSIMD_FLOAT64_C(   606.25), EASYSIMD_FLOAT64_C(   291.08), EASYSIMD_FLOAT64_C(  -706.55),
        EASYSIMD_FLOAT64_C(  -881.59), EASYSIMD_FLOAT64_C(   634.23), EASYSIMD_FLOAT64_C(  -554.48), EASYSIMD_FLOAT64_C(    -2.30) } },
  };

  easysimd__m512d src, a, r;

  src = easysimd_mm512_loadu_pd(test_vec[0].src);
  a = easysimd_mm512_loadu_pd(test_vec[0].a);
  r = easysimd_mm512_mask_roundscale_pd(src, test_vec[0].k, a, INT32_C(         192));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[0].r), 1);

  src = easysimd_mm512_loadu_pd(test_vec[1].src);
  a = easysimd_mm512_loadu_pd(test_vec[1].a);
  r = easysimd_mm512_mask_roundscale_pd(src, test_vec[1].k, a, INT32_C(          81));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[1].r), 1);

  src = easysimd_mm512_loadu_pd(test_vec[2].src);
  a = easysimd_mm512_loadu_pd(test_vec[2].a);
  r = easysimd_mm512_mask_roundscale_pd(src, test_vec[2].k, a, INT32_C(         242));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[2].r), 1);

  src = easysimd_mm512_loadu_pd(test_vec[3].src);
  a = easysimd_mm512_loadu_pd(test_vec[3].a);
  r = easysimd_mm512_mask_roundscale_pd(src, test_vec[3].k, a, INT32_C(          99));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[3].r), 1);

  src = easysimd_mm512_loadu_pd(test_vec[4].src);
  a = easysimd_mm512_loadu_pd(test_vec[4].a);
  r = easysimd_mm512_mask_roundscale_pd(src, test_vec[4].k, a, INT32_C(           4));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[4].r), 1);

  src = easysimd_mm512_loadu_pd(test_vec[5].src);
  a = easysimd_mm512_loadu_pd(test_vec[5].a);
  r = easysimd_mm512_mask_roundscale_pd(src, test_vec[5].k, a, INT32_C(         192));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[5].r), 1);

  src = easysimd_mm512_loadu_pd(test_vec[6].src);
  a = easysimd_mm512_loadu_pd(test_vec[6].a);
  r = easysimd_mm512_mask_roundscale_pd(src, test_vec[6].k, a, INT32_C(         177));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[6].r), 1);

  src = easysimd_mm512_loadu_pd(test_vec[7].src);
  a = easysimd_mm512_loadu_pd(test_vec[7].a);
  r = easysimd_mm512_mask_roundscale_pd(src, test_vec[7].k, a, INT32_C(         178));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512d src = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    if((easysimd_test_codegen_rand() & 1)) {
      if((easysimd_test_codegen_rand() & 1))
        a = easysimd_mm512_mask_mov_pd(a, 1, easysimd_mm512_set1_pd(EASYSIMD_MATH_NAN));
      else {
        if((easysimd_test_codegen_rand() & 1))
          a = easysimd_mm512_mask_mov_pd(a, 2, easysimd_mm512_set1_pd(EASYSIMD_MATH_INFINITY));
        else
          a = easysimd_mm512_mask_mov_pd(a, 2, easysimd_mm512_set1_pd(-EASYSIMD_MATH_INFINITY));
      }
    }
    int imm8 = (((easysimd_test_codegen_rand() & 15) << 4) | round_type[i % 5]) & 255;
    easysimd__m512d r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm512_mask_roundscale_pd, r, easysimd_mm512_setzero_pd(), imm8, src, k, a);

    easysimd_test_x86_write_f64x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_roundscale_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[8];
    const int32_t imm8;
    const easysimd_float64 r[8];
  } test_vec[] = {
    { UINT8_C(216),
      { EASYSIMD_FLOAT64_C(   774.48), EASYSIMD_FLOAT64_C(  -741.37), EASYSIMD_FLOAT64_C(  -683.64), EASYSIMD_FLOAT64_C(  -597.61),
        EASYSIMD_FLOAT64_C(  -197.26), EASYSIMD_FLOAT64_C(   147.89), EASYSIMD_FLOAT64_C(   506.26), EASYSIMD_FLOAT64_C(   -74.98) },
       INT32_C(         160),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -597.61),
        EASYSIMD_FLOAT64_C(  -197.26), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   506.26), EASYSIMD_FLOAT64_C(   -74.98) } },
    { UINT8_C(243),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(    38.53), EASYSIMD_FLOAT64_C(   693.77), EASYSIMD_FLOAT64_C(  -201.30),
        EASYSIMD_FLOAT64_C(   702.42), EASYSIMD_FLOAT64_C(  -122.05), EASYSIMD_FLOAT64_C(   273.37), EASYSIMD_FLOAT64_C(    98.98) },
       INT32_C(         145),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(    38.53), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(   702.42), EASYSIMD_FLOAT64_C(  -122.05), EASYSIMD_FLOAT64_C(   273.37), EASYSIMD_FLOAT64_C(    98.98) } },
    { UINT8_C( 32),
      { EASYSIMD_FLOAT64_C(   832.04), EASYSIMD_FLOAT64_C(  -176.36), EASYSIMD_FLOAT64_C(  -679.40), EASYSIMD_FLOAT64_C(  -722.44),
        EASYSIMD_FLOAT64_C(   821.34), EASYSIMD_FLOAT64_C(   623.31), EASYSIMD_FLOAT64_C(  -296.98), EASYSIMD_FLOAT64_C(     0.12) },
       INT32_C(         242),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   623.31), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 37),
      { EASYSIMD_FLOAT64_C(   800.18), EASYSIMD_FLOAT64_C(   764.40), EASYSIMD_FLOAT64_C(  -535.63), EASYSIMD_FLOAT64_C(   306.44),
        EASYSIMD_FLOAT64_C(  -310.58), EASYSIMD_FLOAT64_C(   631.30), EASYSIMD_FLOAT64_C(   861.33), EASYSIMD_FLOAT64_C(  -563.91) },
       INT32_C(          67),
      { EASYSIMD_FLOAT64_C(   800.12), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -535.62), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   631.25), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(113),
      { EASYSIMD_FLOAT64_C(   482.93),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(     7.81), EASYSIMD_FLOAT64_C(  -243.70),
        EASYSIMD_FLOAT64_C(   701.26), EASYSIMD_FLOAT64_C(  -596.90), EASYSIMD_FLOAT64_C(  -705.10), EASYSIMD_FLOAT64_C(  -593.51) },
       INT32_C(         164),
      { EASYSIMD_FLOAT64_C(   482.93), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(   701.26), EASYSIMD_FLOAT64_C(  -596.90), EASYSIMD_FLOAT64_C(  -705.10), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(137),
      { EASYSIMD_FLOAT64_C(    51.46),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   107.51), EASYSIMD_FLOAT64_C(  -948.42),
        EASYSIMD_FLOAT64_C(  -568.82), EASYSIMD_FLOAT64_C(  -930.83), EASYSIMD_FLOAT64_C(   368.05), EASYSIMD_FLOAT64_C(  -768.64) },
       INT32_C(          64),
      { EASYSIMD_FLOAT64_C(    51.44), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -948.44),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -768.62) } },
    { UINT8_C(245),
      { EASYSIMD_FLOAT64_C(   399.14), EASYSIMD_FLOAT64_C(   -40.92), EASYSIMD_FLOAT64_C(  -852.05), EASYSIMD_FLOAT64_C(  -701.01),
        EASYSIMD_FLOAT64_C(    88.94), EASYSIMD_FLOAT64_C(   630.88), EASYSIMD_FLOAT64_C(   -98.73), EASYSIMD_FLOAT64_C(  -903.25) },
       INT32_C(         225),
      { EASYSIMD_FLOAT64_C(   399.14), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -852.05), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(    88.94), EASYSIMD_FLOAT64_C(   630.88), EASYSIMD_FLOAT64_C(   -98.73), EASYSIMD_FLOAT64_C(  -903.25) } },
    { UINT8_C( 71),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(     9.01), EASYSIMD_FLOAT64_C(   589.32), EASYSIMD_FLOAT64_C(  -190.99),
        EASYSIMD_FLOAT64_C(  -760.86), EASYSIMD_FLOAT64_C(    -0.61), EASYSIMD_FLOAT64_C(   213.50), EASYSIMD_FLOAT64_C(   290.60) },
       INT32_C(         210),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(     9.01), EASYSIMD_FLOAT64_C(   589.32), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   213.50), EASYSIMD_FLOAT64_C(     0.00) } },
  };

  easysimd__m512d a, r;

  a = easysimd_mm512_loadu_pd(test_vec[0].a);
  r = easysimd_mm512_maskz_roundscale_pd(test_vec[0].k, a, INT32_C(         160));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[1].a);
  r = easysimd_mm512_maskz_roundscale_pd(test_vec[1].k, a, INT32_C(         145));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[2].a);
  r = easysimd_mm512_maskz_roundscale_pd(test_vec[2].k, a, INT32_C(         242));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[3].a);
  r = easysimd_mm512_maskz_roundscale_pd(test_vec[3].k, a, INT32_C(          67));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[4].a);
  r = easysimd_mm512_maskz_roundscale_pd(test_vec[4].k, a, INT32_C(         164));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[5].a);
  r = easysimd_mm512_maskz_roundscale_pd(test_vec[5].k, a, INT32_C(          64));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[6].a);
  r = easysimd_mm512_maskz_roundscale_pd(test_vec[6].k, a, INT32_C(         225));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[7].a);
  r = easysimd_mm512_maskz_roundscale_pd(test_vec[7].k, a, INT32_C(         210));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    if((easysimd_test_codegen_rand() & 1)) {
      if((easysimd_test_codegen_rand() & 1))
        a = easysimd_mm512_mask_mov_pd(a, 1, easysimd_mm512_set1_pd(EASYSIMD_MATH_NAN));
      else {
        if((easysimd_test_codegen_rand() & 1))
          a = easysimd_mm512_mask_mov_pd(a, 2, easysimd_mm512_set1_pd(EASYSIMD_MATH_INFINITY));
        else
          a = easysimd_mm512_mask_mov_pd(a, 2, easysimd_mm512_set1_pd(-EASYSIMD_MATH_INFINITY));
      }
    }
    int imm8 = (((easysimd_test_codegen_rand() & 15) << 4) | round_type[i % 5]) & 255;
    easysimd__m512d r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm512_maskz_roundscale_pd, r, easysimd_mm512_setzero_pd(), imm8, k, a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_roundscale_ss (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const int32_t imm8;
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   822.63), EASYSIMD_FLOAT32_C(   440.98), EASYSIMD_FLOAT32_C(    -7.85), EASYSIMD_FLOAT32_C(   646.73) },
      { EASYSIMD_FLOAT32_C(   446.35), EASYSIMD_FLOAT32_C(  -293.17), EASYSIMD_FLOAT32_C(   587.62), EASYSIMD_FLOAT32_C(   860.34) },
       INT32_C(           0),
      { EASYSIMD_FLOAT32_C(   446.00), EASYSIMD_FLOAT32_C(   440.98), EASYSIMD_FLOAT32_C(    -7.85), EASYSIMD_FLOAT32_C(   646.73) } },
    { { EASYSIMD_FLOAT32_C(    14.68), EASYSIMD_FLOAT32_C(   115.83), EASYSIMD_FLOAT32_C(   -68.33), EASYSIMD_FLOAT32_C(   366.87) },
      { EASYSIMD_FLOAT32_C(  -324.24), EASYSIMD_FLOAT32_C(  -488.69), EASYSIMD_FLOAT32_C(   -87.18), EASYSIMD_FLOAT32_C(  -980.26) },
       INT32_C(          16),
      { EASYSIMD_FLOAT32_C(  -324.00), EASYSIMD_FLOAT32_C(   115.83), EASYSIMD_FLOAT32_C(   -68.33), EASYSIMD_FLOAT32_C(   366.87) } },
    { { EASYSIMD_FLOAT32_C(   673.86), EASYSIMD_FLOAT32_C(   884.60), EASYSIMD_FLOAT32_C(   702.77), EASYSIMD_FLOAT32_C(  -321.62) },
      { EASYSIMD_FLOAT32_C(  -887.81), EASYSIMD_FLOAT32_C(   897.13), EASYSIMD_FLOAT32_C(   967.53), EASYSIMD_FLOAT32_C(  -824.56) },
       INT32_C(          32),
      { EASYSIMD_FLOAT32_C(  -887.75), EASYSIMD_FLOAT32_C(   884.60), EASYSIMD_FLOAT32_C(   702.77), EASYSIMD_FLOAT32_C(  -321.62) } },
    { { EASYSIMD_FLOAT32_C(  -444.99), EASYSIMD_FLOAT32_C(   531.19), EASYSIMD_FLOAT32_C(  -158.59), EASYSIMD_FLOAT32_C(   -20.00) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -717.61), EASYSIMD_FLOAT32_C(   972.15), EASYSIMD_FLOAT32_C(     0.55) },
       INT32_C(          48),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   531.19), EASYSIMD_FLOAT32_C(  -158.59), EASYSIMD_FLOAT32_C(   -20.00) } },
    { { EASYSIMD_FLOAT32_C(  -411.83), EASYSIMD_FLOAT32_C(   589.09), EASYSIMD_FLOAT32_C(   577.76), EASYSIMD_FLOAT32_C(   602.85) },
      { EASYSIMD_FLOAT32_C(  -295.09), EASYSIMD_FLOAT32_C(  -490.58), EASYSIMD_FLOAT32_C(   -30.29), EASYSIMD_FLOAT32_C(   380.68) },
       INT32_C(          64),
      { EASYSIMD_FLOAT32_C(  -295.06), EASYSIMD_FLOAT32_C(   589.09), EASYSIMD_FLOAT32_C(   577.76), EASYSIMD_FLOAT32_C(   602.85) } },
    { { EASYSIMD_FLOAT32_C(   882.54), EASYSIMD_FLOAT32_C(   400.42), EASYSIMD_FLOAT32_C(  -726.72), EASYSIMD_FLOAT32_C(   556.40) },
      {      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   976.05), EASYSIMD_FLOAT32_C(  -765.22), EASYSIMD_FLOAT32_C(   397.20) },
       INT32_C(          80),
      {      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   400.42), EASYSIMD_FLOAT32_C(  -726.72), EASYSIMD_FLOAT32_C(   556.40) } },
    { { EASYSIMD_FLOAT32_C(  -105.81), EASYSIMD_FLOAT32_C(  -242.69), EASYSIMD_FLOAT32_C(   103.84), EASYSIMD_FLOAT32_C(   735.60) },
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -542.34), EASYSIMD_FLOAT32_C(  -982.01), EASYSIMD_FLOAT32_C(   709.46) },
       INT32_C(          96),
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -242.69), EASYSIMD_FLOAT32_C(   103.84), EASYSIMD_FLOAT32_C(   735.60) } },
    { { EASYSIMD_FLOAT32_C(  -953.62), EASYSIMD_FLOAT32_C(   335.83), EASYSIMD_FLOAT32_C(   966.20), EASYSIMD_FLOAT32_C(   649.22) },
      { EASYSIMD_FLOAT32_C(  -959.26), EASYSIMD_FLOAT32_C(  -524.38), EASYSIMD_FLOAT32_C(  -381.06), EASYSIMD_FLOAT32_C(   421.42) },
       INT32_C(         112),
      { EASYSIMD_FLOAT32_C(  -959.26), EASYSIMD_FLOAT32_C(   335.83), EASYSIMD_FLOAT32_C(   966.20), EASYSIMD_FLOAT32_C(   649.22) } },
    { { EASYSIMD_FLOAT32_C(  -498.52), EASYSIMD_FLOAT32_C(  -178.17), EASYSIMD_FLOAT32_C(   769.63), EASYSIMD_FLOAT32_C(  -942.12) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   745.68), EASYSIMD_FLOAT32_C(  -707.34), EASYSIMD_FLOAT32_C(   504.05) },
       INT32_C(         128),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -178.17), EASYSIMD_FLOAT32_C(   769.63), EASYSIMD_FLOAT32_C(  -942.12) } },
    { { EASYSIMD_FLOAT32_C(    76.70), EASYSIMD_FLOAT32_C(  -486.95), EASYSIMD_FLOAT32_C(   252.28), EASYSIMD_FLOAT32_C(  -819.47) },
      { EASYSIMD_FLOAT32_C(  -751.35), EASYSIMD_FLOAT32_C(   -10.40), EASYSIMD_FLOAT32_C(  -361.81), EASYSIMD_FLOAT32_C(  -733.36) },
       INT32_C(         144),
      { EASYSIMD_FLOAT32_C(  -751.35), EASYSIMD_FLOAT32_C(  -486.95), EASYSIMD_FLOAT32_C(   252.28), EASYSIMD_FLOAT32_C(  -819.47) } },
    { { EASYSIMD_FLOAT32_C(  -903.60), EASYSIMD_FLOAT32_C(  -986.62), EASYSIMD_FLOAT32_C(    87.50), EASYSIMD_FLOAT32_C(  -857.23) },
      { EASYSIMD_FLOAT32_C(   349.21), EASYSIMD_FLOAT32_C(    53.70), EASYSIMD_FLOAT32_C(   792.00), EASYSIMD_FLOAT32_C(   389.95) },
       INT32_C(         160),
      { EASYSIMD_FLOAT32_C(   349.21), EASYSIMD_FLOAT32_C(  -986.62), EASYSIMD_FLOAT32_C(    87.50), EASYSIMD_FLOAT32_C(  -857.23) } },
    { { EASYSIMD_FLOAT32_C(  -589.06), EASYSIMD_FLOAT32_C(  -188.63), EASYSIMD_FLOAT32_C(    25.67), EASYSIMD_FLOAT32_C(   -87.58) },
      { EASYSIMD_FLOAT32_C(   633.20), EASYSIMD_FLOAT32_C(  -204.71), EASYSIMD_FLOAT32_C(   -29.71), EASYSIMD_FLOAT32_C(   740.05) },
       INT32_C(         176),
      { EASYSIMD_FLOAT32_C(   633.20), EASYSIMD_FLOAT32_C(  -188.63), EASYSIMD_FLOAT32_C(    25.67), EASYSIMD_FLOAT32_C(   -87.58) } },
    { { EASYSIMD_FLOAT32_C(   262.95), EASYSIMD_FLOAT32_C(   244.10), EASYSIMD_FLOAT32_C(  -840.17), EASYSIMD_FLOAT32_C(   757.92) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -327.12), EASYSIMD_FLOAT32_C(    10.20), EASYSIMD_FLOAT32_C(  -498.67) },
       INT32_C(         192),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   244.10), EASYSIMD_FLOAT32_C(  -840.17), EASYSIMD_FLOAT32_C(   757.92) } },
    { { EASYSIMD_FLOAT32_C(   139.52), EASYSIMD_FLOAT32_C(   188.17), EASYSIMD_FLOAT32_C(  -301.15), EASYSIMD_FLOAT32_C(   235.91) },
      { EASYSIMD_FLOAT32_C(   201.55), EASYSIMD_FLOAT32_C(   786.35), EASYSIMD_FLOAT32_C(   378.69), EASYSIMD_FLOAT32_C(  -449.24) },
       INT32_C(         208),
      { EASYSIMD_FLOAT32_C(   201.55), EASYSIMD_FLOAT32_C(   188.17), EASYSIMD_FLOAT32_C(  -301.15), EASYSIMD_FLOAT32_C(   235.91) } },
    { { EASYSIMD_FLOAT32_C(   170.69), EASYSIMD_FLOAT32_C(   940.71), EASYSIMD_FLOAT32_C(  -630.64), EASYSIMD_FLOAT32_C(   581.62) },
      { EASYSIMD_FLOAT32_C(  -247.93), EASYSIMD_FLOAT32_C(   395.03), EASYSIMD_FLOAT32_C(  -505.96), EASYSIMD_FLOAT32_C(  -614.73) },
       INT32_C(         224),
      { EASYSIMD_FLOAT32_C(  -247.93), EASYSIMD_FLOAT32_C(   940.71), EASYSIMD_FLOAT32_C(  -630.64), EASYSIMD_FLOAT32_C(   581.62) } },
    { { EASYSIMD_FLOAT32_C(   464.33), EASYSIMD_FLOAT32_C(  -874.68), EASYSIMD_FLOAT32_C(  -268.70), EASYSIMD_FLOAT32_C(  -272.72) },
      { EASYSIMD_FLOAT32_C(   369.41), EASYSIMD_FLOAT32_C(  -108.87), EASYSIMD_FLOAT32_C(  -514.80), EASYSIMD_FLOAT32_C(   690.21) },
       INT32_C(         240),
      { EASYSIMD_FLOAT32_C(   369.41), EASYSIMD_FLOAT32_C(  -874.68), EASYSIMD_FLOAT32_C(  -268.70), EASYSIMD_FLOAT32_C(  -272.72) } },
    { { EASYSIMD_FLOAT32_C(   495.39), EASYSIMD_FLOAT32_C(  -808.46), EASYSIMD_FLOAT32_C(  -514.46), EASYSIMD_FLOAT32_C(   495.19) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   673.71), EASYSIMD_FLOAT32_C(  -805.96), EASYSIMD_FLOAT32_C(  -433.04) },
       INT32_C(           1),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -808.46), EASYSIMD_FLOAT32_C(  -514.46), EASYSIMD_FLOAT32_C(   495.19) } },
    { { EASYSIMD_FLOAT32_C(   945.65), EASYSIMD_FLOAT32_C(   426.02), EASYSIMD_FLOAT32_C(  -179.56), EASYSIMD_FLOAT32_C(   116.34) },
      {      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   189.80), EASYSIMD_FLOAT32_C(  -302.04), EASYSIMD_FLOAT32_C(  -881.20) },
       INT32_C(          17),
      {      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   426.02), EASYSIMD_FLOAT32_C(  -179.56), EASYSIMD_FLOAT32_C(   116.34) } },
    { { EASYSIMD_FLOAT32_C(  -224.85), EASYSIMD_FLOAT32_C(  -343.67), EASYSIMD_FLOAT32_C(  -370.62), EASYSIMD_FLOAT32_C(   506.44) },
      { EASYSIMD_FLOAT32_C(   383.61), EASYSIMD_FLOAT32_C(   998.80), EASYSIMD_FLOAT32_C(  -602.43), EASYSIMD_FLOAT32_C(   868.80) },
       INT32_C(          33),
      { EASYSIMD_FLOAT32_C(   383.50), EASYSIMD_FLOAT32_C(  -343.67), EASYSIMD_FLOAT32_C(  -370.62), EASYSIMD_FLOAT32_C(   506.44) } },
    { { EASYSIMD_FLOAT32_C(   961.58), EASYSIMD_FLOAT32_C(   364.20), EASYSIMD_FLOAT32_C(   880.54), EASYSIMD_FLOAT32_C(  -552.88) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   211.59), EASYSIMD_FLOAT32_C(  -879.17), EASYSIMD_FLOAT32_C(    53.42) },
       INT32_C(          49),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   364.20), EASYSIMD_FLOAT32_C(   880.54), EASYSIMD_FLOAT32_C(  -552.88) } },
    { { EASYSIMD_FLOAT32_C(    33.81), EASYSIMD_FLOAT32_C(   724.21), EASYSIMD_FLOAT32_C(  -577.89), EASYSIMD_FLOAT32_C(   854.25) },
      {      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   788.83), EASYSIMD_FLOAT32_C(    44.04), EASYSIMD_FLOAT32_C(   538.50) },
       INT32_C(          65),
      {      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   724.21), EASYSIMD_FLOAT32_C(  -577.89), EASYSIMD_FLOAT32_C(   854.25) } },
    { { EASYSIMD_FLOAT32_C(  -588.30), EASYSIMD_FLOAT32_C(  -595.98), EASYSIMD_FLOAT32_C(   386.83), EASYSIMD_FLOAT32_C(    41.08) },
      { EASYSIMD_FLOAT32_C(   910.46), EASYSIMD_FLOAT32_C(  -229.56), EASYSIMD_FLOAT32_C(    39.88), EASYSIMD_FLOAT32_C(  -691.97) },
       INT32_C(          81),
      { EASYSIMD_FLOAT32_C(   910.44), EASYSIMD_FLOAT32_C(  -595.98), EASYSIMD_FLOAT32_C(   386.83), EASYSIMD_FLOAT32_C(    41.08) } },
    { { EASYSIMD_FLOAT32_C(  -271.12), EASYSIMD_FLOAT32_C(  -730.39), EASYSIMD_FLOAT32_C(  -996.56), EASYSIMD_FLOAT32_C(  -390.58) },
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -137.18), EASYSIMD_FLOAT32_C(   821.01), EASYSIMD_FLOAT32_C(  -162.44) },
       INT32_C(          97),
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -730.39), EASYSIMD_FLOAT32_C(  -996.56), EASYSIMD_FLOAT32_C(  -390.58) } },
    { { EASYSIMD_FLOAT32_C(   -49.95), EASYSIMD_FLOAT32_C(   323.78), EASYSIMD_FLOAT32_C(  -744.24), EASYSIMD_FLOAT32_C(  -195.70) },
      { EASYSIMD_FLOAT32_C(  -835.68), EASYSIMD_FLOAT32_C(  -955.41), EASYSIMD_FLOAT32_C(   848.34), EASYSIMD_FLOAT32_C(   702.82) },
       INT32_C(         113),
      { EASYSIMD_FLOAT32_C(  -835.69), EASYSIMD_FLOAT32_C(   323.78), EASYSIMD_FLOAT32_C(  -744.24), EASYSIMD_FLOAT32_C(  -195.70) } },
    { { EASYSIMD_FLOAT32_C(   477.21), EASYSIMD_FLOAT32_C(  -566.68), EASYSIMD_FLOAT32_C(  -636.08), EASYSIMD_FLOAT32_C(   881.23) },
      { EASYSIMD_FLOAT32_C(   820.15), EASYSIMD_FLOAT32_C(   405.00), EASYSIMD_FLOAT32_C(   791.69), EASYSIMD_FLOAT32_C(  -409.41) },
       INT32_C(         129),
      { EASYSIMD_FLOAT32_C(   820.15), EASYSIMD_FLOAT32_C(  -566.68), EASYSIMD_FLOAT32_C(  -636.08), EASYSIMD_FLOAT32_C(   881.23) } },
    { { EASYSIMD_FLOAT32_C(  -900.28), EASYSIMD_FLOAT32_C(   229.83), EASYSIMD_FLOAT32_C(   173.76), EASYSIMD_FLOAT32_C(  -630.67) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   783.18), EASYSIMD_FLOAT32_C(    86.06), EASYSIMD_FLOAT32_C(  -903.91) },
       INT32_C(         145),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   229.83), EASYSIMD_FLOAT32_C(   173.76), EASYSIMD_FLOAT32_C(  -630.67) } },
    { { EASYSIMD_FLOAT32_C(  -987.67), EASYSIMD_FLOAT32_C(   203.76), EASYSIMD_FLOAT32_C(   757.27), EASYSIMD_FLOAT32_C(   -37.61) },
      {      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -986.97), EASYSIMD_FLOAT32_C(   766.68), EASYSIMD_FLOAT32_C(  -308.15) },
       INT32_C(         161),
      {      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   203.76), EASYSIMD_FLOAT32_C(   757.27), EASYSIMD_FLOAT32_C(   -37.61) } },
    { { EASYSIMD_FLOAT32_C(  -990.16), EASYSIMD_FLOAT32_C(    92.24), EASYSIMD_FLOAT32_C(  -172.00), EASYSIMD_FLOAT32_C(  -626.25) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -351.85), EASYSIMD_FLOAT32_C(   778.75), EASYSIMD_FLOAT32_C(  -234.84) },
       INT32_C(         177),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    92.24), EASYSIMD_FLOAT32_C(  -172.00), EASYSIMD_FLOAT32_C(  -626.25) } },
    { { EASYSIMD_FLOAT32_C(  -135.13), EASYSIMD_FLOAT32_C(  -531.43), EASYSIMD_FLOAT32_C(   397.38), EASYSIMD_FLOAT32_C(   234.20) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   180.56), EASYSIMD_FLOAT32_C(  -679.74), EASYSIMD_FLOAT32_C(   797.93) },
       INT32_C(         193),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -531.43), EASYSIMD_FLOAT32_C(   397.38), EASYSIMD_FLOAT32_C(   234.20) } },
    { { EASYSIMD_FLOAT32_C(   810.27), EASYSIMD_FLOAT32_C(   988.51), EASYSIMD_FLOAT32_C(  -998.85), EASYSIMD_FLOAT32_C(  -227.35) },
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -985.82), EASYSIMD_FLOAT32_C(  -460.66), EASYSIMD_FLOAT32_C(   207.90) },
       INT32_C(         209),
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   988.51), EASYSIMD_FLOAT32_C(  -998.85), EASYSIMD_FLOAT32_C(  -227.35) } },
    { { EASYSIMD_FLOAT32_C(  -918.36), EASYSIMD_FLOAT32_C(   246.61), EASYSIMD_FLOAT32_C(  -569.43), EASYSIMD_FLOAT32_C(  -544.61) },
      { EASYSIMD_FLOAT32_C(  -779.92), EASYSIMD_FLOAT32_C(    78.72), EASYSIMD_FLOAT32_C(  -765.86), EASYSIMD_FLOAT32_C(   -14.77) },
       INT32_C(         225),
      { EASYSIMD_FLOAT32_C(  -779.92), EASYSIMD_FLOAT32_C(   246.61), EASYSIMD_FLOAT32_C(  -569.43), EASYSIMD_FLOAT32_C(  -544.61) } },
    { { EASYSIMD_FLOAT32_C(  -542.23), EASYSIMD_FLOAT32_C(   850.11), EASYSIMD_FLOAT32_C(  -213.97), EASYSIMD_FLOAT32_C(   855.15) },
      { EASYSIMD_FLOAT32_C(    84.31), EASYSIMD_FLOAT32_C(  -512.14), EASYSIMD_FLOAT32_C(    35.71), EASYSIMD_FLOAT32_C(   404.57) },
       INT32_C(         241),
      { EASYSIMD_FLOAT32_C(    84.31), EASYSIMD_FLOAT32_C(   850.11), EASYSIMD_FLOAT32_C(  -213.97), EASYSIMD_FLOAT32_C(   855.15) } },
    { { EASYSIMD_FLOAT32_C(   820.46), EASYSIMD_FLOAT32_C(   648.45), EASYSIMD_FLOAT32_C(  -903.94), EASYSIMD_FLOAT32_C(   808.97) },
      { EASYSIMD_FLOAT32_C(   649.61), EASYSIMD_FLOAT32_C(  -131.28), EASYSIMD_FLOAT32_C(  -674.98), EASYSIMD_FLOAT32_C(   663.79) },
       INT32_C(           2),
      { EASYSIMD_FLOAT32_C(   650.00), EASYSIMD_FLOAT32_C(   648.45), EASYSIMD_FLOAT32_C(  -903.94), EASYSIMD_FLOAT32_C(   808.97) } },
    { { EASYSIMD_FLOAT32_C(   532.92), EASYSIMD_FLOAT32_C(   735.59), EASYSIMD_FLOAT32_C(   562.42), EASYSIMD_FLOAT32_C(   135.50) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -190.97), EASYSIMD_FLOAT32_C(   566.07), EASYSIMD_FLOAT32_C(  -727.38) },
       INT32_C(          18),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   735.59), EASYSIMD_FLOAT32_C(   562.42), EASYSIMD_FLOAT32_C(   135.50) } },
    { { EASYSIMD_FLOAT32_C(  -493.23), EASYSIMD_FLOAT32_C(  -985.67), EASYSIMD_FLOAT32_C(   -37.75), EASYSIMD_FLOAT32_C(   -35.47) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   748.28), EASYSIMD_FLOAT32_C(  -180.32), EASYSIMD_FLOAT32_C(   -51.25) },
       INT32_C(          34),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -985.67), EASYSIMD_FLOAT32_C(   -37.75), EASYSIMD_FLOAT32_C(   -35.47) } },
    { { EASYSIMD_FLOAT32_C(  -646.68), EASYSIMD_FLOAT32_C(  -478.07), EASYSIMD_FLOAT32_C(   675.84), EASYSIMD_FLOAT32_C(  -998.23) },
      { EASYSIMD_FLOAT32_C(  -382.00), EASYSIMD_FLOAT32_C(   484.81), EASYSIMD_FLOAT32_C(   651.37), EASYSIMD_FLOAT32_C(   486.71) },
       INT32_C(          50),
      { EASYSIMD_FLOAT32_C(  -382.00), EASYSIMD_FLOAT32_C(  -478.07), EASYSIMD_FLOAT32_C(   675.84), EASYSIMD_FLOAT32_C(  -998.23) } },
    { { EASYSIMD_FLOAT32_C(   315.16), EASYSIMD_FLOAT32_C(  -105.23), EASYSIMD_FLOAT32_C(   342.75), EASYSIMD_FLOAT32_C(    50.75) },
      { EASYSIMD_FLOAT32_C(  -542.82), EASYSIMD_FLOAT32_C(  -521.75), EASYSIMD_FLOAT32_C(  -132.02), EASYSIMD_FLOAT32_C(   266.21) },
       INT32_C(          66),
      { EASYSIMD_FLOAT32_C(  -542.81), EASYSIMD_FLOAT32_C(  -105.23), EASYSIMD_FLOAT32_C(   342.75), EASYSIMD_FLOAT32_C(    50.75) } },
    { { EASYSIMD_FLOAT32_C(   140.61), EASYSIMD_FLOAT32_C(  -704.69), EASYSIMD_FLOAT32_C(  -310.89), EASYSIMD_FLOAT32_C(   647.37) },
      {      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   651.36), EASYSIMD_FLOAT32_C(  -388.09), EASYSIMD_FLOAT32_C(  -825.91) },
       INT32_C(          82),
      {      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -704.69), EASYSIMD_FLOAT32_C(  -310.89), EASYSIMD_FLOAT32_C(   647.37) } },
    { { EASYSIMD_FLOAT32_C(   635.78), EASYSIMD_FLOAT32_C(   286.97), EASYSIMD_FLOAT32_C(   476.15), EASYSIMD_FLOAT32_C(  -842.29) },
      { EASYSIMD_FLOAT32_C(   -37.18), EASYSIMD_FLOAT32_C(   477.92), EASYSIMD_FLOAT32_C(  -224.29), EASYSIMD_FLOAT32_C(  -552.37) },
       INT32_C(          98),
      { EASYSIMD_FLOAT32_C(   -37.17), EASYSIMD_FLOAT32_C(   286.97), EASYSIMD_FLOAT32_C(   476.15), EASYSIMD_FLOAT32_C(  -842.29) } },
    { { EASYSIMD_FLOAT32_C(  -737.58), EASYSIMD_FLOAT32_C(  -742.54), EASYSIMD_FLOAT32_C(  -555.54), EASYSIMD_FLOAT32_C(   157.18) },
      { EASYSIMD_FLOAT32_C(   600.21), EASYSIMD_FLOAT32_C(   495.21), EASYSIMD_FLOAT32_C(   614.37), EASYSIMD_FLOAT32_C(  -921.55) },
       INT32_C(         114),
      { EASYSIMD_FLOAT32_C(   600.21), EASYSIMD_FLOAT32_C(  -742.54), EASYSIMD_FLOAT32_C(  -555.54), EASYSIMD_FLOAT32_C(   157.18) } },
    { { EASYSIMD_FLOAT32_C(  -119.42), EASYSIMD_FLOAT32_C(  -877.23), EASYSIMD_FLOAT32_C(   503.80), EASYSIMD_FLOAT32_C(   175.89) },
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   151.17), EASYSIMD_FLOAT32_C(   485.54), EASYSIMD_FLOAT32_C(  -536.76) },
       INT32_C(         130),
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -877.23), EASYSIMD_FLOAT32_C(   503.80), EASYSIMD_FLOAT32_C(   175.89) } },
    { { EASYSIMD_FLOAT32_C(   194.66), EASYSIMD_FLOAT32_C(  -217.55), EASYSIMD_FLOAT32_C(   498.65), EASYSIMD_FLOAT32_C(  -518.36) },
      { EASYSIMD_FLOAT32_C(  -741.39), EASYSIMD_FLOAT32_C(   656.36), EASYSIMD_FLOAT32_C(   444.45), EASYSIMD_FLOAT32_C(   736.52) },
       INT32_C(         146),
      { EASYSIMD_FLOAT32_C(  -741.39), EASYSIMD_FLOAT32_C(  -217.55), EASYSIMD_FLOAT32_C(   498.65), EASYSIMD_FLOAT32_C(  -518.36) } },
    { { EASYSIMD_FLOAT32_C(   892.08), EASYSIMD_FLOAT32_C(  -134.18), EASYSIMD_FLOAT32_C(  -305.51), EASYSIMD_FLOAT32_C(  -850.46) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   851.67), EASYSIMD_FLOAT32_C(   749.75), EASYSIMD_FLOAT32_C(  -194.51) },
       INT32_C(         162),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -134.18), EASYSIMD_FLOAT32_C(  -305.51), EASYSIMD_FLOAT32_C(  -850.46) } },
    { { EASYSIMD_FLOAT32_C(   168.68), EASYSIMD_FLOAT32_C(  -653.38), EASYSIMD_FLOAT32_C(   950.97), EASYSIMD_FLOAT32_C(  -327.52) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -237.14), EASYSIMD_FLOAT32_C(   823.66), EASYSIMD_FLOAT32_C(     8.05) },
       INT32_C(         178),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -653.38), EASYSIMD_FLOAT32_C(   950.97), EASYSIMD_FLOAT32_C(  -327.52) } },
    { { EASYSIMD_FLOAT32_C(  -332.33), EASYSIMD_FLOAT32_C(    88.98), EASYSIMD_FLOAT32_C(  -218.60), EASYSIMD_FLOAT32_C(   450.13) },
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   263.04), EASYSIMD_FLOAT32_C(   708.73), EASYSIMD_FLOAT32_C(  -756.01) },
       INT32_C(         194),
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(    88.98), EASYSIMD_FLOAT32_C(  -218.60), EASYSIMD_FLOAT32_C(   450.13) } },
    { { EASYSIMD_FLOAT32_C(  -400.43), EASYSIMD_FLOAT32_C(  -688.92), EASYSIMD_FLOAT32_C(   370.55), EASYSIMD_FLOAT32_C(  -250.89) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   222.23), EASYSIMD_FLOAT32_C(  -501.14), EASYSIMD_FLOAT32_C(  -573.16) },
       INT32_C(         210),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -688.92), EASYSIMD_FLOAT32_C(   370.55), EASYSIMD_FLOAT32_C(  -250.89) } },
    { { EASYSIMD_FLOAT32_C(   595.53), EASYSIMD_FLOAT32_C(    34.89), EASYSIMD_FLOAT32_C(  -721.97), EASYSIMD_FLOAT32_C(  -731.99) },
      { EASYSIMD_FLOAT32_C(  -442.59), EASYSIMD_FLOAT32_C(    40.89), EASYSIMD_FLOAT32_C(  -908.33), EASYSIMD_FLOAT32_C(   565.46) },
       INT32_C(         226),
      { EASYSIMD_FLOAT32_C(  -442.59), EASYSIMD_FLOAT32_C(    34.89), EASYSIMD_FLOAT32_C(  -721.97), EASYSIMD_FLOAT32_C(  -731.99) } },
    { { EASYSIMD_FLOAT32_C(   678.40), EASYSIMD_FLOAT32_C(  -766.87), EASYSIMD_FLOAT32_C(   355.96), EASYSIMD_FLOAT32_C(  -540.20) },
      { EASYSIMD_FLOAT32_C(   683.26), EASYSIMD_FLOAT32_C(   943.59), EASYSIMD_FLOAT32_C(   722.84), EASYSIMD_FLOAT32_C(   391.99) },
       INT32_C(         242),
      { EASYSIMD_FLOAT32_C(   683.26), EASYSIMD_FLOAT32_C(  -766.87), EASYSIMD_FLOAT32_C(   355.96), EASYSIMD_FLOAT32_C(  -540.20) } },
    { { EASYSIMD_FLOAT32_C(  -569.67), EASYSIMD_FLOAT32_C(  -162.75), EASYSIMD_FLOAT32_C(  -136.35), EASYSIMD_FLOAT32_C(    29.90) },
      { EASYSIMD_FLOAT32_C(   148.32), EASYSIMD_FLOAT32_C(  -765.79), EASYSIMD_FLOAT32_C(   779.01), EASYSIMD_FLOAT32_C(  -230.32) },
       INT32_C(           3),
      { EASYSIMD_FLOAT32_C(   148.00), EASYSIMD_FLOAT32_C(  -162.75), EASYSIMD_FLOAT32_C(  -136.35), EASYSIMD_FLOAT32_C(    29.90) } },
    { { EASYSIMD_FLOAT32_C(  -722.13), EASYSIMD_FLOAT32_C(   196.52), EASYSIMD_FLOAT32_C(  -855.30), EASYSIMD_FLOAT32_C(  -395.08) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   179.60), EASYSIMD_FLOAT32_C(  -117.04), EASYSIMD_FLOAT32_C(    60.06) },
       INT32_C(          19),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   196.52), EASYSIMD_FLOAT32_C(  -855.30), EASYSIMD_FLOAT32_C(  -395.08) } },
    { { EASYSIMD_FLOAT32_C(   151.73), EASYSIMD_FLOAT32_C(   302.46), EASYSIMD_FLOAT32_C(  -809.17), EASYSIMD_FLOAT32_C(  -169.87) },
      {      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   546.80), EASYSIMD_FLOAT32_C(   289.94), EASYSIMD_FLOAT32_C(   218.85) },
       INT32_C(          35),
      {      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   302.46), EASYSIMD_FLOAT32_C(  -809.17), EASYSIMD_FLOAT32_C(  -169.87) } },
    { { EASYSIMD_FLOAT32_C(   677.98), EASYSIMD_FLOAT32_C(   443.10), EASYSIMD_FLOAT32_C(   448.09), EASYSIMD_FLOAT32_C(  -458.37) },
      { EASYSIMD_FLOAT32_C(  -527.00), EASYSIMD_FLOAT32_C(  -403.59), EASYSIMD_FLOAT32_C(  -224.16), EASYSIMD_FLOAT32_C(  -748.00) },
       INT32_C(          51),
      { EASYSIMD_FLOAT32_C(  -527.00), EASYSIMD_FLOAT32_C(   443.10), EASYSIMD_FLOAT32_C(   448.09), EASYSIMD_FLOAT32_C(  -458.37) } },
    { { EASYSIMD_FLOAT32_C(  -767.73), EASYSIMD_FLOAT32_C(  -470.13), EASYSIMD_FLOAT32_C(  -437.39), EASYSIMD_FLOAT32_C(  -623.03) },
      { EASYSIMD_FLOAT32_C(   134.79), EASYSIMD_FLOAT32_C(   354.66), EASYSIMD_FLOAT32_C(   556.57), EASYSIMD_FLOAT32_C(  -982.25) },
       INT32_C(          67),
      { EASYSIMD_FLOAT32_C(   134.75), EASYSIMD_FLOAT32_C(  -470.13), EASYSIMD_FLOAT32_C(  -437.39), EASYSIMD_FLOAT32_C(  -623.03) } },
    { { EASYSIMD_FLOAT32_C(   293.57), EASYSIMD_FLOAT32_C(   941.60), EASYSIMD_FLOAT32_C(   566.45), EASYSIMD_FLOAT32_C(  -403.97) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -603.41), EASYSIMD_FLOAT32_C(  -868.37), EASYSIMD_FLOAT32_C(   679.23) },
       INT32_C(          83),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   941.60), EASYSIMD_FLOAT32_C(   566.45), EASYSIMD_FLOAT32_C(  -403.97) } },
    { { EASYSIMD_FLOAT32_C(   169.62), EASYSIMD_FLOAT32_C(  -300.70), EASYSIMD_FLOAT32_C(   961.31), EASYSIMD_FLOAT32_C(  -152.40) },
      { EASYSIMD_FLOAT32_C(  -857.60), EASYSIMD_FLOAT32_C(   409.40), EASYSIMD_FLOAT32_C(   389.23), EASYSIMD_FLOAT32_C(  -384.61) },
       INT32_C(          99),
      { EASYSIMD_FLOAT32_C(  -857.59), EASYSIMD_FLOAT32_C(  -300.70), EASYSIMD_FLOAT32_C(   961.31), EASYSIMD_FLOAT32_C(  -152.40) } },
    { { EASYSIMD_FLOAT32_C(  -834.93), EASYSIMD_FLOAT32_C(  -132.61), EASYSIMD_FLOAT32_C(   371.90), EASYSIMD_FLOAT32_C(  -602.66) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   934.51), EASYSIMD_FLOAT32_C(  -225.69), EASYSIMD_FLOAT32_C(  -467.95) },
       INT32_C(         115),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -132.61), EASYSIMD_FLOAT32_C(   371.90), EASYSIMD_FLOAT32_C(  -602.66) } },
    { { EASYSIMD_FLOAT32_C(  -450.20), EASYSIMD_FLOAT32_C(   703.89), EASYSIMD_FLOAT32_C(   624.45), EASYSIMD_FLOAT32_C(  -508.60) },
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -779.51), EASYSIMD_FLOAT32_C(  -376.17), EASYSIMD_FLOAT32_C(   666.94) },
       INT32_C(         131),
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   703.89), EASYSIMD_FLOAT32_C(   624.45), EASYSIMD_FLOAT32_C(  -508.60) } },
    { { EASYSIMD_FLOAT32_C(   702.59), EASYSIMD_FLOAT32_C(   472.68), EASYSIMD_FLOAT32_C(  -947.24), EASYSIMD_FLOAT32_C(   663.90) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -804.85), EASYSIMD_FLOAT32_C(    73.30), EASYSIMD_FLOAT32_C(   709.52) },
       INT32_C(         147),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   472.68), EASYSIMD_FLOAT32_C(  -947.24), EASYSIMD_FLOAT32_C(   663.90) } },
    { { EASYSIMD_FLOAT32_C(   874.59), EASYSIMD_FLOAT32_C(   677.94), EASYSIMD_FLOAT32_C(  -548.99), EASYSIMD_FLOAT32_C(  -728.07) },
      {      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -614.48), EASYSIMD_FLOAT32_C(    46.24), EASYSIMD_FLOAT32_C(   607.25) },
       INT32_C(         163),
      {      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   677.94), EASYSIMD_FLOAT32_C(  -548.99), EASYSIMD_FLOAT32_C(  -728.07) } },
    { { EASYSIMD_FLOAT32_C(   378.59), EASYSIMD_FLOAT32_C(     1.58), EASYSIMD_FLOAT32_C(  -351.54), EASYSIMD_FLOAT32_C(  -351.06) },
      { EASYSIMD_FLOAT32_C(   222.07), EASYSIMD_FLOAT32_C(   272.29), EASYSIMD_FLOAT32_C(  -684.12), EASYSIMD_FLOAT32_C(   574.18) },
       INT32_C(         179),
      { EASYSIMD_FLOAT32_C(   222.07), EASYSIMD_FLOAT32_C(     1.58), EASYSIMD_FLOAT32_C(  -351.54), EASYSIMD_FLOAT32_C(  -351.06) } },
    { { EASYSIMD_FLOAT32_C(   669.34), EASYSIMD_FLOAT32_C(   276.77), EASYSIMD_FLOAT32_C(    48.04), EASYSIMD_FLOAT32_C(   722.09) },
      { EASYSIMD_FLOAT32_C(   -59.33), EASYSIMD_FLOAT32_C(   368.32), EASYSIMD_FLOAT32_C(   917.25), EASYSIMD_FLOAT32_C(  -986.02) },
       INT32_C(         195),
      { EASYSIMD_FLOAT32_C(   -59.33), EASYSIMD_FLOAT32_C(   276.77), EASYSIMD_FLOAT32_C(    48.04), EASYSIMD_FLOAT32_C(   722.09) } },
    { { EASYSIMD_FLOAT32_C(  -272.21), EASYSIMD_FLOAT32_C(    93.09), EASYSIMD_FLOAT32_C(   -47.57), EASYSIMD_FLOAT32_C(  -594.27) },
      { EASYSIMD_FLOAT32_C(   544.11), EASYSIMD_FLOAT32_C(   224.35), EASYSIMD_FLOAT32_C(   480.93), EASYSIMD_FLOAT32_C(   929.63) },
       INT32_C(         211),
      { EASYSIMD_FLOAT32_C(   544.11), EASYSIMD_FLOAT32_C(    93.09), EASYSIMD_FLOAT32_C(   -47.57), EASYSIMD_FLOAT32_C(  -594.27) } },
    { { EASYSIMD_FLOAT32_C(    88.18), EASYSIMD_FLOAT32_C(   604.33), EASYSIMD_FLOAT32_C(   647.72), EASYSIMD_FLOAT32_C(   245.24) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -350.71), EASYSIMD_FLOAT32_C(   893.70), EASYSIMD_FLOAT32_C(   631.86) },
       INT32_C(         227),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   604.33), EASYSIMD_FLOAT32_C(   647.72), EASYSIMD_FLOAT32_C(   245.24) } },
    { { EASYSIMD_FLOAT32_C(   947.73), EASYSIMD_FLOAT32_C(   445.54), EASYSIMD_FLOAT32_C(  -258.65), EASYSIMD_FLOAT32_C(   617.07) },
      { EASYSIMD_FLOAT32_C(  -277.69), EASYSIMD_FLOAT32_C(   789.38), EASYSIMD_FLOAT32_C(   339.16), EASYSIMD_FLOAT32_C(   662.98) },
       INT32_C(         243),
      { EASYSIMD_FLOAT32_C(  -277.69), EASYSIMD_FLOAT32_C(   445.54), EASYSIMD_FLOAT32_C(  -258.65), EASYSIMD_FLOAT32_C(   617.07) } },
    { { EASYSIMD_FLOAT32_C(   256.41), EASYSIMD_FLOAT32_C(   676.96), EASYSIMD_FLOAT32_C(  -764.46), EASYSIMD_FLOAT32_C(   984.20) },
      { EASYSIMD_FLOAT32_C(  -229.94), EASYSIMD_FLOAT32_C(   187.97), EASYSIMD_FLOAT32_C(  -610.07), EASYSIMD_FLOAT32_C(  -685.83) },
       INT32_C(           4),
      { EASYSIMD_FLOAT32_C(  -230.00), EASYSIMD_FLOAT32_C(   676.96), EASYSIMD_FLOAT32_C(  -764.46), EASYSIMD_FLOAT32_C(   984.20) } },
    { { EASYSIMD_FLOAT32_C(   870.86), EASYSIMD_FLOAT32_C(  -756.20), EASYSIMD_FLOAT32_C(  -317.08), EASYSIMD_FLOAT32_C(   -40.96) },
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -669.37), EASYSIMD_FLOAT32_C(  -795.72), EASYSIMD_FLOAT32_C(  -168.95) },
       INT32_C(          20),
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -756.20), EASYSIMD_FLOAT32_C(  -317.08), EASYSIMD_FLOAT32_C(   -40.96) } },
    { { EASYSIMD_FLOAT32_C(  -148.72), EASYSIMD_FLOAT32_C(   263.97), EASYSIMD_FLOAT32_C(  -589.36), EASYSIMD_FLOAT32_C(  -703.18) },
      { EASYSIMD_FLOAT32_C(  -994.68), EASYSIMD_FLOAT32_C(  -972.29), EASYSIMD_FLOAT32_C(    19.13), EASYSIMD_FLOAT32_C(   794.70) },
       INT32_C(          36),
      { EASYSIMD_FLOAT32_C(  -994.75), EASYSIMD_FLOAT32_C(   263.97), EASYSIMD_FLOAT32_C(  -589.36), EASYSIMD_FLOAT32_C(  -703.18) } },
    { { EASYSIMD_FLOAT32_C(  -317.88), EASYSIMD_FLOAT32_C(   -47.60), EASYSIMD_FLOAT32_C(  -376.72), EASYSIMD_FLOAT32_C(  -640.92) },
      { EASYSIMD_FLOAT32_C(   187.95), EASYSIMD_FLOAT32_C(  -392.52), EASYSIMD_FLOAT32_C(   129.14), EASYSIMD_FLOAT32_C(  -624.08) },
       INT32_C(          52),
      { EASYSIMD_FLOAT32_C(   188.00), EASYSIMD_FLOAT32_C(   -47.60), EASYSIMD_FLOAT32_C(  -376.72), EASYSIMD_FLOAT32_C(  -640.92) } },
    { { EASYSIMD_FLOAT32_C(   443.31), EASYSIMD_FLOAT32_C(  -211.76), EASYSIMD_FLOAT32_C(  -131.74), EASYSIMD_FLOAT32_C(   687.10) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   827.30), EASYSIMD_FLOAT32_C(   535.23), EASYSIMD_FLOAT32_C(   801.79) },
       INT32_C(          68),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -211.76), EASYSIMD_FLOAT32_C(  -131.74), EASYSIMD_FLOAT32_C(   687.10) } },
    { { EASYSIMD_FLOAT32_C(  -218.29), EASYSIMD_FLOAT32_C(  -870.44), EASYSIMD_FLOAT32_C(  -170.82), EASYSIMD_FLOAT32_C(   633.00) },
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   239.82), EASYSIMD_FLOAT32_C(   929.82), EASYSIMD_FLOAT32_C(   398.84) },
       INT32_C(          84),
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -870.44), EASYSIMD_FLOAT32_C(  -170.82), EASYSIMD_FLOAT32_C(   633.00) } },
    { { EASYSIMD_FLOAT32_C(  -365.60), EASYSIMD_FLOAT32_C(   631.08), EASYSIMD_FLOAT32_C(  -854.05), EASYSIMD_FLOAT32_C(   257.67) },
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   333.90), EASYSIMD_FLOAT32_C(   865.15), EASYSIMD_FLOAT32_C(   119.30) },
       INT32_C(         100),
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   631.08), EASYSIMD_FLOAT32_C(  -854.05), EASYSIMD_FLOAT32_C(   257.67) } },
    { { EASYSIMD_FLOAT32_C(  -501.95), EASYSIMD_FLOAT32_C(   730.83), EASYSIMD_FLOAT32_C(  -750.29), EASYSIMD_FLOAT32_C(   969.21) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   784.94), EASYSIMD_FLOAT32_C(   771.00), EASYSIMD_FLOAT32_C(   589.71) },
       INT32_C(         116),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   730.83), EASYSIMD_FLOAT32_C(  -750.29), EASYSIMD_FLOAT32_C(   969.21) } },
    { { EASYSIMD_FLOAT32_C(   719.27), EASYSIMD_FLOAT32_C(   -19.60), EASYSIMD_FLOAT32_C(  -814.29), EASYSIMD_FLOAT32_C(   112.80) },
      { EASYSIMD_FLOAT32_C(  -779.77), EASYSIMD_FLOAT32_C(  -884.47), EASYSIMD_FLOAT32_C(  -488.36), EASYSIMD_FLOAT32_C(   487.75) },
       INT32_C(         132),
      { EASYSIMD_FLOAT32_C(  -779.77), EASYSIMD_FLOAT32_C(   -19.60), EASYSIMD_FLOAT32_C(  -814.29), EASYSIMD_FLOAT32_C(   112.80) } },
    { { EASYSIMD_FLOAT32_C(   705.18), EASYSIMD_FLOAT32_C(  -877.85), EASYSIMD_FLOAT32_C(  -304.44), EASYSIMD_FLOAT32_C(   851.13) },
      { EASYSIMD_FLOAT32_C(   379.82), EASYSIMD_FLOAT32_C(  -314.28), EASYSIMD_FLOAT32_C(   185.03), EASYSIMD_FLOAT32_C(   244.98) },
       INT32_C(         148),
      { EASYSIMD_FLOAT32_C(   379.82), EASYSIMD_FLOAT32_C(  -877.85), EASYSIMD_FLOAT32_C(  -304.44), EASYSIMD_FLOAT32_C(   851.13) } },
    { { EASYSIMD_FLOAT32_C(  -105.16), EASYSIMD_FLOAT32_C(  -892.46), EASYSIMD_FLOAT32_C(  -632.37), EASYSIMD_FLOAT32_C(   392.89) },
      {      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -382.66), EASYSIMD_FLOAT32_C(   362.10), EASYSIMD_FLOAT32_C(   396.50) },
       INT32_C(         164),
      {      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -892.46), EASYSIMD_FLOAT32_C(  -632.37), EASYSIMD_FLOAT32_C(   392.89) } },
    { { EASYSIMD_FLOAT32_C(  -446.50), EASYSIMD_FLOAT32_C(   685.81), EASYSIMD_FLOAT32_C(  -294.53), EASYSIMD_FLOAT32_C(   533.91) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   818.27), EASYSIMD_FLOAT32_C(   754.13), EASYSIMD_FLOAT32_C(   987.04) },
       INT32_C(         180),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   685.81), EASYSIMD_FLOAT32_C(  -294.53), EASYSIMD_FLOAT32_C(   533.91) } },
    { { EASYSIMD_FLOAT32_C(    51.52), EASYSIMD_FLOAT32_C(  -964.91), EASYSIMD_FLOAT32_C(   364.03), EASYSIMD_FLOAT32_C(   747.08) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -256.14), EASYSIMD_FLOAT32_C(  -567.20), EASYSIMD_FLOAT32_C(    71.25) },
       INT32_C(         196),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -964.91), EASYSIMD_FLOAT32_C(   364.03), EASYSIMD_FLOAT32_C(   747.08) } },
    { { EASYSIMD_FLOAT32_C(   966.09), EASYSIMD_FLOAT32_C(  -903.63), EASYSIMD_FLOAT32_C(  -394.56), EASYSIMD_FLOAT32_C(   358.98) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   222.78), EASYSIMD_FLOAT32_C(  -278.91), EASYSIMD_FLOAT32_C(   331.23) },
       INT32_C(         212),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -903.63), EASYSIMD_FLOAT32_C(  -394.56), EASYSIMD_FLOAT32_C(   358.98) } },
    { { EASYSIMD_FLOAT32_C(  -682.56), EASYSIMD_FLOAT32_C(  -821.44), EASYSIMD_FLOAT32_C(   539.99), EASYSIMD_FLOAT32_C(    22.92) },
      { EASYSIMD_FLOAT32_C(   712.47), EASYSIMD_FLOAT32_C(   411.50), EASYSIMD_FLOAT32_C(  -158.81), EASYSIMD_FLOAT32_C(   466.60) },
       INT32_C(         228),
      { EASYSIMD_FLOAT32_C(   712.47), EASYSIMD_FLOAT32_C(  -821.44), EASYSIMD_FLOAT32_C(   539.99), EASYSIMD_FLOAT32_C(    22.92) } },
    { { EASYSIMD_FLOAT32_C(   171.10), EASYSIMD_FLOAT32_C(  -291.52), EASYSIMD_FLOAT32_C(  -549.93), EASYSIMD_FLOAT32_C(   206.19) },
      { EASYSIMD_FLOAT32_C(  -927.48), EASYSIMD_FLOAT32_C(  -802.85), EASYSIMD_FLOAT32_C(    92.41), EASYSIMD_FLOAT32_C(  -183.62) },
       INT32_C(         244),
      { EASYSIMD_FLOAT32_C(  -927.48), EASYSIMD_FLOAT32_C(  -291.52), EASYSIMD_FLOAT32_C(  -549.93), EASYSIMD_FLOAT32_C(   206.19) } },
  };

  easysimd__m128 a, b, r;

  a = easysimd_mm_loadu_ps(test_vec[0].a);
  b = easysimd_mm_loadu_ps(test_vec[0].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(           0));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[1].a);
  b = easysimd_mm_loadu_ps(test_vec[1].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          16));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[2].a);
  b = easysimd_mm_loadu_ps(test_vec[2].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          32));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[3].a);
  b = easysimd_mm_loadu_ps(test_vec[3].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          48));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[4].a);
  b = easysimd_mm_loadu_ps(test_vec[4].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          64));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[5].a);
  b = easysimd_mm_loadu_ps(test_vec[5].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          80));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[6].a);
  b = easysimd_mm_loadu_ps(test_vec[6].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          96));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[7].a);
  b = easysimd_mm_loadu_ps(test_vec[7].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         112));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[7].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[8].a);
  b = easysimd_mm_loadu_ps(test_vec[8].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         128));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[8].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[9].a);
  b = easysimd_mm_loadu_ps(test_vec[9].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         144));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[9].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[10].a);
  b = easysimd_mm_loadu_ps(test_vec[10].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         160));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[10].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[11].a);
  b = easysimd_mm_loadu_ps(test_vec[11].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         176));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[11].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[12].a);
  b = easysimd_mm_loadu_ps(test_vec[12].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         192));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[12].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[13].a);
  b = easysimd_mm_loadu_ps(test_vec[13].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         208));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[13].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[14].a);
  b = easysimd_mm_loadu_ps(test_vec[14].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         224));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[14].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[15].a);
  b = easysimd_mm_loadu_ps(test_vec[15].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         240));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[15].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[16].a);
  b = easysimd_mm_loadu_ps(test_vec[16].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(           1));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[16].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[17].a);
  b = easysimd_mm_loadu_ps(test_vec[17].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          17));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[17].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[18].a);
  b = easysimd_mm_loadu_ps(test_vec[18].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          33));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[18].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[19].a);
  b = easysimd_mm_loadu_ps(test_vec[19].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          49));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[19].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[20].a);
  b = easysimd_mm_loadu_ps(test_vec[20].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          65));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[20].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[21].a);
  b = easysimd_mm_loadu_ps(test_vec[21].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          81));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[21].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[22].a);
  b = easysimd_mm_loadu_ps(test_vec[22].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          97));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[22].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[23].a);
  b = easysimd_mm_loadu_ps(test_vec[23].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         113));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[23].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[24].a);
  b = easysimd_mm_loadu_ps(test_vec[24].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         129));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[24].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[25].a);
  b = easysimd_mm_loadu_ps(test_vec[25].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         145));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[25].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[26].a);
  b = easysimd_mm_loadu_ps(test_vec[26].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         161));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[26].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[27].a);
  b = easysimd_mm_loadu_ps(test_vec[27].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         177));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[27].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[28].a);
  b = easysimd_mm_loadu_ps(test_vec[28].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         193));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[28].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[29].a);
  b = easysimd_mm_loadu_ps(test_vec[29].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         209));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[29].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[30].a);
  b = easysimd_mm_loadu_ps(test_vec[30].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         225));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[30].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[31].a);
  b = easysimd_mm_loadu_ps(test_vec[31].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         241));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[31].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[32].a);
  b = easysimd_mm_loadu_ps(test_vec[32].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(           2));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[32].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[33].a);
  b = easysimd_mm_loadu_ps(test_vec[33].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          18));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[33].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[34].a);
  b = easysimd_mm_loadu_ps(test_vec[34].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          34));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[34].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[35].a);
  b = easysimd_mm_loadu_ps(test_vec[35].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          50));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[35].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[36].a);
  b = easysimd_mm_loadu_ps(test_vec[36].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          66));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[36].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[37].a);
  b = easysimd_mm_loadu_ps(test_vec[37].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          82));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[37].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[38].a);
  b = easysimd_mm_loadu_ps(test_vec[38].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          98));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[38].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[39].a);
  b = easysimd_mm_loadu_ps(test_vec[39].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         114));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[39].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[40].a);
  b = easysimd_mm_loadu_ps(test_vec[40].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         130));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[40].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[41].a);
  b = easysimd_mm_loadu_ps(test_vec[41].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         146));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[41].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[42].a);
  b = easysimd_mm_loadu_ps(test_vec[42].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         162));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[42].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[43].a);
  b = easysimd_mm_loadu_ps(test_vec[43].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         178));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[43].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[44].a);
  b = easysimd_mm_loadu_ps(test_vec[44].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         194));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[44].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[45].a);
  b = easysimd_mm_loadu_ps(test_vec[45].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         210));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[45].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[46].a);
  b = easysimd_mm_loadu_ps(test_vec[46].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         226));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[46].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[47].a);
  b = easysimd_mm_loadu_ps(test_vec[47].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         242));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[47].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[48].a);
  b = easysimd_mm_loadu_ps(test_vec[48].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(           3));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[48].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[49].a);
  b = easysimd_mm_loadu_ps(test_vec[49].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          19));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[49].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[50].a);
  b = easysimd_mm_loadu_ps(test_vec[50].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          35));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[50].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[51].a);
  b = easysimd_mm_loadu_ps(test_vec[51].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          51));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[51].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[52].a);
  b = easysimd_mm_loadu_ps(test_vec[52].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          67));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[52].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[53].a);
  b = easysimd_mm_loadu_ps(test_vec[53].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          83));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[53].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[54].a);
  b = easysimd_mm_loadu_ps(test_vec[54].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          99));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[54].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[55].a);
  b = easysimd_mm_loadu_ps(test_vec[55].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         115));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[55].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[56].a);
  b = easysimd_mm_loadu_ps(test_vec[56].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         131));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[56].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[57].a);
  b = easysimd_mm_loadu_ps(test_vec[57].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         147));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[57].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[58].a);
  b = easysimd_mm_loadu_ps(test_vec[58].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         163));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[58].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[59].a);
  b = easysimd_mm_loadu_ps(test_vec[59].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         179));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[59].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[60].a);
  b = easysimd_mm_loadu_ps(test_vec[60].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         195));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[60].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[61].a);
  b = easysimd_mm_loadu_ps(test_vec[61].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         211));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[61].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[62].a);
  b = easysimd_mm_loadu_ps(test_vec[62].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         227));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[62].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[63].a);
  b = easysimd_mm_loadu_ps(test_vec[63].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         243));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[63].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[64].a);
  b = easysimd_mm_loadu_ps(test_vec[64].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(           4));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[64].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[65].a);
  b = easysimd_mm_loadu_ps(test_vec[65].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          20));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[65].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[66].a);
  b = easysimd_mm_loadu_ps(test_vec[66].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          36));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[66].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[67].a);
  b = easysimd_mm_loadu_ps(test_vec[67].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          52));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[67].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[68].a);
  b = easysimd_mm_loadu_ps(test_vec[68].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          68));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[68].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[69].a);
  b = easysimd_mm_loadu_ps(test_vec[69].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(          84));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[69].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[70].a);
  b = easysimd_mm_loadu_ps(test_vec[70].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         100));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[70].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[71].a);
  b = easysimd_mm_loadu_ps(test_vec[71].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         116));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[71].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[72].a);
  b = easysimd_mm_loadu_ps(test_vec[72].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         132));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[72].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[73].a);
  b = easysimd_mm_loadu_ps(test_vec[73].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         148));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[73].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[74].a);
  b = easysimd_mm_loadu_ps(test_vec[74].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         164));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[74].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[75].a);
  b = easysimd_mm_loadu_ps(test_vec[75].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         180));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[75].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[76].a);
  b = easysimd_mm_loadu_ps(test_vec[76].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         196));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[76].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[77].a);
  b = easysimd_mm_loadu_ps(test_vec[77].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         212));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[77].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[78].a);
  b = easysimd_mm_loadu_ps(test_vec[78].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         228));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[78].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[79].a);
  b = easysimd_mm_loadu_ps(test_vec[79].b);
  r = easysimd_mm_roundscale_ss(a, b, INT32_C(         244));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[79].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 5 ; i++) {
    for (int j = 0 ; j < 16 ; j++) {
      easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
      easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
      if((easysimd_test_codegen_rand() & 1)) {
        if((easysimd_test_codegen_rand() & 1))
          b = easysimd_mm_mask_mov_ps(b, 1, easysimd_mm_set1_ps(EASYSIMD_MATH_NAN));
        else {
          if((easysimd_test_codegen_rand() & 1))
            b = easysimd_mm_mask_mov_ps(b, 1, easysimd_mm_set1_ps(EASYSIMD_MATH_INFINITY));
          else
            b = easysimd_mm_mask_mov_ps(b, 1, easysimd_mm_set1_ps(-EASYSIMD_MATH_INFINITY));
        }
      }
      int imm8 = ((j << 4) | round_type[i]) & 255;
      easysimd__m128 r;
      EASYSIMD_CONSTIFY_256_(easysimd_mm_roundscale_ss, r, easysimd_mm_setzero_ps(), imm8, a, b);

      easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
      easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
      easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
      easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
    }
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_roundscale_ss (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[4];
    const easysimd__mmask8 k;
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const int32_t imm8;
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -866.00), EASYSIMD_FLOAT32_C(  -361.57), EASYSIMD_FLOAT32_C(  -516.74), EASYSIMD_FLOAT32_C(  -299.17) },
      UINT8_C(118),
      { EASYSIMD_FLOAT32_C(  -454.59), EASYSIMD_FLOAT32_C(   290.73), EASYSIMD_FLOAT32_C(   910.53), EASYSIMD_FLOAT32_C(    91.54) },
      { EASYSIMD_FLOAT32_C(   463.34), EASYSIMD_FLOAT32_C(   465.84), EASYSIMD_FLOAT32_C(  -211.43), EASYSIMD_FLOAT32_C(   429.04) },
       INT32_C(          96),
      { EASYSIMD_FLOAT32_C(  -866.00), EASYSIMD_FLOAT32_C(   290.73), EASYSIMD_FLOAT32_C(   910.53), EASYSIMD_FLOAT32_C(    91.54) } },
    { { EASYSIMD_FLOAT32_C(   762.35), EASYSIMD_FLOAT32_C(   -43.33), EASYSIMD_FLOAT32_C(  -810.77), EASYSIMD_FLOAT32_C(  -340.20) },
      UINT8_C(132),
      { EASYSIMD_FLOAT32_C(   433.68), EASYSIMD_FLOAT32_C(  -841.73), EASYSIMD_FLOAT32_C(  -334.46), EASYSIMD_FLOAT32_C(  -927.21) },
      { EASYSIMD_FLOAT32_C(   555.23), EASYSIMD_FLOAT32_C(   678.38), EASYSIMD_FLOAT32_C(   943.43), EASYSIMD_FLOAT32_C(  -947.24) },
       INT32_C(          17),
      { EASYSIMD_FLOAT32_C(   762.35), EASYSIMD_FLOAT32_C(  -841.73), EASYSIMD_FLOAT32_C(  -334.46), EASYSIMD_FLOAT32_C(  -927.21) } },
    { { EASYSIMD_FLOAT32_C(   523.07), EASYSIMD_FLOAT32_C(  -723.81), EASYSIMD_FLOAT32_C(   581.28), EASYSIMD_FLOAT32_C(  -993.68) },
      UINT8_C(145),
      { EASYSIMD_FLOAT32_C(  -438.97), EASYSIMD_FLOAT32_C(  -448.27), EASYSIMD_FLOAT32_C(  -732.25), EASYSIMD_FLOAT32_C(  -528.44) },
      { EASYSIMD_FLOAT32_C(   643.27), EASYSIMD_FLOAT32_C(   731.10), EASYSIMD_FLOAT32_C(   937.40), EASYSIMD_FLOAT32_C(  -568.16) },
       INT32_C(         130),
      { EASYSIMD_FLOAT32_C(   643.27), EASYSIMD_FLOAT32_C(  -448.27), EASYSIMD_FLOAT32_C(  -732.25), EASYSIMD_FLOAT32_C(  -528.44) } },
    { { EASYSIMD_FLOAT32_C(  -341.66), EASYSIMD_FLOAT32_C(   -77.52), EASYSIMD_FLOAT32_C(  -879.37), EASYSIMD_FLOAT32_C(  -152.44) },
      UINT8_C(247),
      { EASYSIMD_FLOAT32_C(   701.96), EASYSIMD_FLOAT32_C(  -718.75), EASYSIMD_FLOAT32_C(   740.54), EASYSIMD_FLOAT32_C(  -632.50) },
      { EASYSIMD_FLOAT32_C(  -645.96), EASYSIMD_FLOAT32_C(   295.77), EASYSIMD_FLOAT32_C(  -954.12), EASYSIMD_FLOAT32_C(  -702.53) },
       INT32_C(         227),
      { EASYSIMD_FLOAT32_C(  -645.96), EASYSIMD_FLOAT32_C(  -718.75), EASYSIMD_FLOAT32_C(   740.54), EASYSIMD_FLOAT32_C(  -632.50) } },
    { { EASYSIMD_FLOAT32_C(   240.32), EASYSIMD_FLOAT32_C(  -128.41), EASYSIMD_FLOAT32_C(  -535.73), EASYSIMD_FLOAT32_C(  -178.40) },
      UINT8_C( 78),
      { EASYSIMD_FLOAT32_C(   441.29), EASYSIMD_FLOAT32_C(   382.63), EASYSIMD_FLOAT32_C(   429.65), EASYSIMD_FLOAT32_C(   709.04) },
      { EASYSIMD_FLOAT32_C(   854.19), EASYSIMD_FLOAT32_C(    72.92), EASYSIMD_FLOAT32_C(   440.14), EASYSIMD_FLOAT32_C(   791.59) },
       INT32_C(         100),
      { EASYSIMD_FLOAT32_C(   240.32), EASYSIMD_FLOAT32_C(   382.63), EASYSIMD_FLOAT32_C(   429.65), EASYSIMD_FLOAT32_C(   709.04) } },
    { { EASYSIMD_FLOAT32_C(   -44.45), EASYSIMD_FLOAT32_C(  -836.91), EASYSIMD_FLOAT32_C(   522.76), EASYSIMD_FLOAT32_C(    76.18) },
      UINT8_C(115),
      { EASYSIMD_FLOAT32_C(   105.03), EASYSIMD_FLOAT32_C(  -221.86), EASYSIMD_FLOAT32_C(   291.90), EASYSIMD_FLOAT32_C(  -154.42) },
      { EASYSIMD_FLOAT32_C(   145.64), EASYSIMD_FLOAT32_C(   645.93), EASYSIMD_FLOAT32_C(  -858.66), EASYSIMD_FLOAT32_C(   191.53) },
       INT32_C(          48),
      { EASYSIMD_FLOAT32_C(   145.62), EASYSIMD_FLOAT32_C(  -221.86), EASYSIMD_FLOAT32_C(   291.90), EASYSIMD_FLOAT32_C(  -154.42) } },
    { { EASYSIMD_FLOAT32_C(   379.60), EASYSIMD_FLOAT32_C(   183.72), EASYSIMD_FLOAT32_C(  -638.53), EASYSIMD_FLOAT32_C(   843.87) },
      UINT8_C( 95),
      { EASYSIMD_FLOAT32_C(   239.38), EASYSIMD_FLOAT32_C(   285.16), EASYSIMD_FLOAT32_C(   387.96), EASYSIMD_FLOAT32_C(  -330.97) },
      { EASYSIMD_FLOAT32_C(    -5.80), EASYSIMD_FLOAT32_C(   242.15), EASYSIMD_FLOAT32_C(   741.95), EASYSIMD_FLOAT32_C(  -565.66) },
       INT32_C(          49),
      { EASYSIMD_FLOAT32_C(    -5.88), EASYSIMD_FLOAT32_C(   285.16), EASYSIMD_FLOAT32_C(   387.96), EASYSIMD_FLOAT32_C(  -330.97) } },
    { { EASYSIMD_FLOAT32_C(    34.62), EASYSIMD_FLOAT32_C(   989.29), EASYSIMD_FLOAT32_C(   409.79), EASYSIMD_FLOAT32_C(  -442.63) },
      UINT8_C(135),
      { EASYSIMD_FLOAT32_C(  -579.56), EASYSIMD_FLOAT32_C(   662.41), EASYSIMD_FLOAT32_C(   843.61), EASYSIMD_FLOAT32_C(   712.34) },
      { EASYSIMD_FLOAT32_C(  -492.02), EASYSIMD_FLOAT32_C(   -10.75), EASYSIMD_FLOAT32_C(   358.27), EASYSIMD_FLOAT32_C(  -350.67) },
       INT32_C(         194),
      { EASYSIMD_FLOAT32_C(  -492.02), EASYSIMD_FLOAT32_C(   662.41), EASYSIMD_FLOAT32_C(   843.61), EASYSIMD_FLOAT32_C(   712.34) } },
  };

  easysimd__m128 src, a, b, r;

  src = easysimd_mm_loadu_ps(test_vec[0].src);
  a = easysimd_mm_loadu_ps(test_vec[0].a);
  b = easysimd_mm_loadu_ps(test_vec[0].b);
  r = easysimd_mm_mask_roundscale_ss(src, test_vec[0].k, a, b, INT32_C(          96));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[0].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[1].src);
  a = easysimd_mm_loadu_ps(test_vec[1].a);
  b = easysimd_mm_loadu_ps(test_vec[1].b);
  r = easysimd_mm_mask_roundscale_ss(src, test_vec[1].k, a, b, INT32_C(          17));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[1].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[2].src);
  a = easysimd_mm_loadu_ps(test_vec[2].a);
  b = easysimd_mm_loadu_ps(test_vec[2].b);
  r = easysimd_mm_mask_roundscale_ss(src, test_vec[2].k, a, b, INT32_C(         130));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[2].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[3].src);
  a = easysimd_mm_loadu_ps(test_vec[3].a);
  b = easysimd_mm_loadu_ps(test_vec[3].b);
  r = easysimd_mm_mask_roundscale_ss(src, test_vec[3].k, a, b, INT32_C(         227));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[3].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[4].src);
  a = easysimd_mm_loadu_ps(test_vec[4].a);
  b = easysimd_mm_loadu_ps(test_vec[4].b);
  r = easysimd_mm_mask_roundscale_ss(src, test_vec[4].k, a, b, INT32_C(         100));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[4].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[5].src);
  a = easysimd_mm_loadu_ps(test_vec[5].a);
  b = easysimd_mm_loadu_ps(test_vec[5].b);
  r = easysimd_mm_mask_roundscale_ss(src, test_vec[5].k, a, b, INT32_C(          48));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[5].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[6].src);
  a = easysimd_mm_loadu_ps(test_vec[6].a);
  b = easysimd_mm_loadu_ps(test_vec[6].b);
  r = easysimd_mm_mask_roundscale_ss(src, test_vec[6].k, a, b, INT32_C(          49));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[6].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[7].src);
  a = easysimd_mm_loadu_ps(test_vec[7].a);
  b = easysimd_mm_loadu_ps(test_vec[7].b);
  r = easysimd_mm_mask_roundscale_ss(src, test_vec[7].k, a, b, INT32_C(         194));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    if((easysimd_test_codegen_rand() & 1)) {
      if((easysimd_test_codegen_rand() & 1))
        b = easysimd_mm_mask_mov_ps(b, 1, easysimd_mm_set1_ps(EASYSIMD_MATH_NAN));
      else {
        if((easysimd_test_codegen_rand() & 1))
          b = easysimd_mm_mask_mov_ps(b, 1, easysimd_mm_set1_ps(EASYSIMD_MATH_INFINITY));
        else
          b = easysimd_mm_mask_mov_ps(b, 1, easysimd_mm_set1_ps(-EASYSIMD_MATH_INFINITY));
      }
    }
    int imm8 = (((easysimd_test_codegen_rand() & 15) << 4) | round_type[i % 5]) & 255;
    easysimd__m128 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm_mask_roundscale_ss, r, easysimd_mm_setzero_ps(), imm8, src, k, a, b);

    easysimd_test_x86_write_f32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_roundscale_ss (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const int32_t imm8;
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C(150),
      { EASYSIMD_FLOAT32_C(   560.38), EASYSIMD_FLOAT32_C(  -514.60), EASYSIMD_FLOAT32_C(  -499.34), EASYSIMD_FLOAT32_C(   404.24) },
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   740.05), EASYSIMD_FLOAT32_C(  -310.60), EASYSIMD_FLOAT32_C(   878.68) },
       INT32_C(         224),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -514.60), EASYSIMD_FLOAT32_C(  -499.34), EASYSIMD_FLOAT32_C(   404.24) } },
    { UINT8_C(137),
      { EASYSIMD_FLOAT32_C(  -845.43), EASYSIMD_FLOAT32_C(   397.73), EASYSIMD_FLOAT32_C(   152.56), EASYSIMD_FLOAT32_C(  -856.14) },
      { EASYSIMD_FLOAT32_C(  -192.48), EASYSIMD_FLOAT32_C(   709.93), EASYSIMD_FLOAT32_C(   209.33), EASYSIMD_FLOAT32_C(   227.96) },
       INT32_C(         129),
      { EASYSIMD_FLOAT32_C(  -192.48), EASYSIMD_FLOAT32_C(   397.73), EASYSIMD_FLOAT32_C(   152.56), EASYSIMD_FLOAT32_C(  -856.14) } },
    { UINT8_C( 20),
      { EASYSIMD_FLOAT32_C(   880.32), EASYSIMD_FLOAT32_C(  -957.81), EASYSIMD_FLOAT32_C(  -701.42), EASYSIMD_FLOAT32_C(  -470.36) },
      { EASYSIMD_FLOAT32_C(  -777.04), EASYSIMD_FLOAT32_C(   600.26), EASYSIMD_FLOAT32_C(  -331.16), EASYSIMD_FLOAT32_C(   783.34) },
       INT32_C(          50),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -957.81), EASYSIMD_FLOAT32_C(  -701.42), EASYSIMD_FLOAT32_C(  -470.36) } },
    { UINT8_C( 12),
      { EASYSIMD_FLOAT32_C(  -423.61), EASYSIMD_FLOAT32_C(   -90.45), EASYSIMD_FLOAT32_C(   876.98), EASYSIMD_FLOAT32_C(  -544.93) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   560.58), EASYSIMD_FLOAT32_C(   575.90), EASYSIMD_FLOAT32_C(   469.65) },
       INT32_C(          99),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -90.45), EASYSIMD_FLOAT32_C(   876.98), EASYSIMD_FLOAT32_C(  -544.93) } },
    { UINT8_C( 12),
      { EASYSIMD_FLOAT32_C(   874.33), EASYSIMD_FLOAT32_C(   674.90), EASYSIMD_FLOAT32_C(  -458.99), EASYSIMD_FLOAT32_C(    83.66) },
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   913.35), EASYSIMD_FLOAT32_C(  -863.41), EASYSIMD_FLOAT32_C(   843.16) },
       INT32_C(         116),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   674.90), EASYSIMD_FLOAT32_C(  -458.99), EASYSIMD_FLOAT32_C(    83.66) } },
    { UINT8_C( 25),
      { EASYSIMD_FLOAT32_C(  -598.26), EASYSIMD_FLOAT32_C(   741.99), EASYSIMD_FLOAT32_C(    -7.85), EASYSIMD_FLOAT32_C(  -814.92) },
      { EASYSIMD_FLOAT32_C(   827.65), EASYSIMD_FLOAT32_C(  -838.35), EASYSIMD_FLOAT32_C(   372.66), EASYSIMD_FLOAT32_C(  -595.96) },
       INT32_C(          96),
      { EASYSIMD_FLOAT32_C(   827.66), EASYSIMD_FLOAT32_C(   741.99), EASYSIMD_FLOAT32_C(    -7.85), EASYSIMD_FLOAT32_C(  -814.92) } },
    { UINT8_C(213),
      { EASYSIMD_FLOAT32_C(  -610.18), EASYSIMD_FLOAT32_C(  -189.78), EASYSIMD_FLOAT32_C(  -564.99), EASYSIMD_FLOAT32_C(   859.47) },
      { EASYSIMD_FLOAT32_C(  -511.25), EASYSIMD_FLOAT32_C(  -834.52), EASYSIMD_FLOAT32_C(  -273.15), EASYSIMD_FLOAT32_C(   319.83) },
       INT32_C(          65),
      { EASYSIMD_FLOAT32_C(  -511.25), EASYSIMD_FLOAT32_C(  -189.78), EASYSIMD_FLOAT32_C(  -564.99), EASYSIMD_FLOAT32_C(   859.47) } },
    { UINT8_C(140),
      { EASYSIMD_FLOAT32_C(   123.48), EASYSIMD_FLOAT32_C(   304.61), EASYSIMD_FLOAT32_C(   774.19), EASYSIMD_FLOAT32_C(   260.07) },
      { EASYSIMD_FLOAT32_C(   147.77), EASYSIMD_FLOAT32_C(   567.85), EASYSIMD_FLOAT32_C(   438.85), EASYSIMD_FLOAT32_C(   289.51) },
       INT32_C(         114),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   304.61), EASYSIMD_FLOAT32_C(   774.19), EASYSIMD_FLOAT32_C(   260.07) } },
  };

  easysimd__m128 a, b, r;

  a = easysimd_mm_loadu_ps(test_vec[0].a);
  b = easysimd_mm_loadu_ps(test_vec[0].b);
  r = easysimd_mm_maskz_roundscale_ss(test_vec[0].k, a, b, INT32_C(         224));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[1].a);
  b = easysimd_mm_loadu_ps(test_vec[1].b);
  r = easysimd_mm_maskz_roundscale_ss(test_vec[1].k, a, b, INT32_C(         129));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[2].a);
  b = easysimd_mm_loadu_ps(test_vec[2].b);
  r = easysimd_mm_maskz_roundscale_ss(test_vec[2].k, a, b, INT32_C(          50));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[3].a);
  b = easysimd_mm_loadu_ps(test_vec[3].b);
  r = easysimd_mm_maskz_roundscale_ss(test_vec[3].k, a, b, INT32_C(          99));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[4].a);
  b = easysimd_mm_loadu_ps(test_vec[4].b);
  r = easysimd_mm_maskz_roundscale_ss(test_vec[4].k, a, b, INT32_C(         116));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[5].a);
  b = easysimd_mm_loadu_ps(test_vec[5].b);
  r = easysimd_mm_maskz_roundscale_ss(test_vec[5].k, a, b, INT32_C(          96));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[6].a);
  b = easysimd_mm_loadu_ps(test_vec[6].b);
  r = easysimd_mm_maskz_roundscale_ss(test_vec[6].k, a, b, INT32_C(          65));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[7].a);
  b = easysimd_mm_loadu_ps(test_vec[7].b);
  r = easysimd_mm_maskz_roundscale_ss(test_vec[7].k, a, b, INT32_C(         114));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    if((easysimd_test_codegen_rand() & 1)) {
      if((easysimd_test_codegen_rand() & 1))
        b = easysimd_mm_mask_mov_ps(b, 1, easysimd_mm_set1_ps(EASYSIMD_MATH_NAN));
      else {
        if((easysimd_test_codegen_rand() & 1))
          b = easysimd_mm_mask_mov_ps(b, 1, easysimd_mm_set1_ps(EASYSIMD_MATH_INFINITY));
        else
          b = easysimd_mm_mask_mov_ps(b, 1, easysimd_mm_set1_ps(-EASYSIMD_MATH_INFINITY));
      }
    }
    int imm8 = (((easysimd_test_codegen_rand() & 15) << 4) | round_type[i % 5]) & 255;
    easysimd__m128 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm_maskz_roundscale_ss, r, easysimd_mm_setzero_ps(), imm8, k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_roundscale_sd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const int32_t imm8;
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   538.74), EASYSIMD_FLOAT64_C(    64.09) },
      { EASYSIMD_FLOAT64_C(   791.27), EASYSIMD_FLOAT64_C(  -766.95) },
       INT32_C(           0),
      { EASYSIMD_FLOAT64_C(   791.00), EASYSIMD_FLOAT64_C(    64.09) } },
    { { EASYSIMD_FLOAT64_C(  -678.97), EASYSIMD_FLOAT64_C(    39.26) },
      { EASYSIMD_FLOAT64_C(  -965.12), EASYSIMD_FLOAT64_C(   198.74) },
       INT32_C(          16),
      { EASYSIMD_FLOAT64_C(  -965.00), EASYSIMD_FLOAT64_C(    39.26) } },
    { { EASYSIMD_FLOAT64_C(   569.29), EASYSIMD_FLOAT64_C(   -63.11) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   227.96) },
       INT32_C(          32),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   -63.11) } },
    { { EASYSIMD_FLOAT64_C(   570.63), EASYSIMD_FLOAT64_C(  -299.33) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   106.40) },
       INT32_C(          48),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -299.33) } },
    { { EASYSIMD_FLOAT64_C(   206.46), EASYSIMD_FLOAT64_C(  -483.51) },
      { EASYSIMD_FLOAT64_C(  -173.23), EASYSIMD_FLOAT64_C(   300.40) },
       INT32_C(          64),
      { EASYSIMD_FLOAT64_C(  -173.25), EASYSIMD_FLOAT64_C(  -483.51) } },
    { { EASYSIMD_FLOAT64_C(   -29.64), EASYSIMD_FLOAT64_C(    87.23) },
      { EASYSIMD_FLOAT64_C(    38.91), EASYSIMD_FLOAT64_C(  -744.22) },
       INT32_C(          80),
      { EASYSIMD_FLOAT64_C(    38.91), EASYSIMD_FLOAT64_C(    87.23) } },
    { { EASYSIMD_FLOAT64_C(  -897.00), EASYSIMD_FLOAT64_C(  -952.95) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -896.36) },
       INT32_C(          96),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -952.95) } },
    { { EASYSIMD_FLOAT64_C(  -861.49), EASYSIMD_FLOAT64_C(   566.82) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   707.81) },
       INT32_C(         112),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   566.82) } },
    { { EASYSIMD_FLOAT64_C(   -64.23), EASYSIMD_FLOAT64_C(   759.82) },
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -493.60) },
       INT32_C(         128),
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   759.82) } },
    { { EASYSIMD_FLOAT64_C(   -48.69), EASYSIMD_FLOAT64_C(  -284.86) },
      { EASYSIMD_FLOAT64_C(  -180.74), EASYSIMD_FLOAT64_C(   467.80) },
       INT32_C(         144),
      { EASYSIMD_FLOAT64_C(  -180.74), EASYSIMD_FLOAT64_C(  -284.86) } },
    { { EASYSIMD_FLOAT64_C(  -880.34), EASYSIMD_FLOAT64_C(  -225.76) },
      { EASYSIMD_FLOAT64_C(  -487.74), EASYSIMD_FLOAT64_C(   206.89) },
       INT32_C(         160),
      { EASYSIMD_FLOAT64_C(  -487.74), EASYSIMD_FLOAT64_C(  -225.76) } },
    { { EASYSIMD_FLOAT64_C(  -231.96), EASYSIMD_FLOAT64_C(   832.86) },
      { EASYSIMD_FLOAT64_C(   916.15), EASYSIMD_FLOAT64_C(  -184.91) },
       INT32_C(         176),
      { EASYSIMD_FLOAT64_C(   916.15), EASYSIMD_FLOAT64_C(   832.86) } },
    { { EASYSIMD_FLOAT64_C(  -980.22), EASYSIMD_FLOAT64_C(   183.17) },
      { EASYSIMD_FLOAT64_C(  -409.84), EASYSIMD_FLOAT64_C(  -841.70) },
       INT32_C(         192),
      { EASYSIMD_FLOAT64_C(  -409.84), EASYSIMD_FLOAT64_C(   183.17) } },
    { { EASYSIMD_FLOAT64_C(   300.74), EASYSIMD_FLOAT64_C(   866.11) },
      { EASYSIMD_FLOAT64_C(   253.70), EASYSIMD_FLOAT64_C(  -117.14) },
       INT32_C(         208),
      { EASYSIMD_FLOAT64_C(   253.70), EASYSIMD_FLOAT64_C(   866.11) } },
    { { EASYSIMD_FLOAT64_C(    13.51), EASYSIMD_FLOAT64_C(  -600.29) },
      { EASYSIMD_FLOAT64_C(   308.28), EASYSIMD_FLOAT64_C(   474.00) },
       INT32_C(         224),
      { EASYSIMD_FLOAT64_C(   308.28), EASYSIMD_FLOAT64_C(  -600.29) } },
    { { EASYSIMD_FLOAT64_C(   -78.92), EASYSIMD_FLOAT64_C(  -574.69) },
      { EASYSIMD_FLOAT64_C(   327.13), EASYSIMD_FLOAT64_C(   740.34) },
       INT32_C(         240),
      { EASYSIMD_FLOAT64_C(   327.13), EASYSIMD_FLOAT64_C(  -574.69) } },
    { { EASYSIMD_FLOAT64_C(  -130.97), EASYSIMD_FLOAT64_C(   860.00) },
      { EASYSIMD_FLOAT64_C(  -332.65), EASYSIMD_FLOAT64_C(   381.30) },
       INT32_C(           1),
      { EASYSIMD_FLOAT64_C(  -333.00), EASYSIMD_FLOAT64_C(   860.00) } },
    { { EASYSIMD_FLOAT64_C(  -519.50), EASYSIMD_FLOAT64_C(  -850.66) },
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -603.35) },
       INT32_C(          17),
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -850.66) } },
    { { EASYSIMD_FLOAT64_C(  -852.40), EASYSIMD_FLOAT64_C(  -818.22) },
      { EASYSIMD_FLOAT64_C(  -425.27), EASYSIMD_FLOAT64_C(  -102.41) },
       INT32_C(          33),
      { EASYSIMD_FLOAT64_C(  -425.50), EASYSIMD_FLOAT64_C(  -818.22) } },
    { { EASYSIMD_FLOAT64_C(  -559.16), EASYSIMD_FLOAT64_C(  -848.72) },
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   242.72) },
       INT32_C(          49),
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -848.72) } },
    { { EASYSIMD_FLOAT64_C(  -361.20), EASYSIMD_FLOAT64_C(   377.06) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(    64.10) },
       INT32_C(          65),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   377.06) } },
    { { EASYSIMD_FLOAT64_C(   -42.79), EASYSIMD_FLOAT64_C(   573.22) },
      { EASYSIMD_FLOAT64_C(    72.42), EASYSIMD_FLOAT64_C(   624.56) },
       INT32_C(          81),
      { EASYSIMD_FLOAT64_C(    72.41), EASYSIMD_FLOAT64_C(   573.22) } },
    { { EASYSIMD_FLOAT64_C(  -860.70), EASYSIMD_FLOAT64_C(  -894.95) },
      { EASYSIMD_FLOAT64_C(   103.87), EASYSIMD_FLOAT64_C(    39.04) },
       INT32_C(          97),
      { EASYSIMD_FLOAT64_C(   103.86), EASYSIMD_FLOAT64_C(  -894.95) } },
    { { EASYSIMD_FLOAT64_C(  -931.70), EASYSIMD_FLOAT64_C(  -369.34) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -784.10) },
       INT32_C(         113),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -369.34) } },
    { { EASYSIMD_FLOAT64_C(   113.49), EASYSIMD_FLOAT64_C(  -705.05) },
      { EASYSIMD_FLOAT64_C(   933.70), EASYSIMD_FLOAT64_C(   264.77) },
       INT32_C(         129),
      { EASYSIMD_FLOAT64_C(   933.70), EASYSIMD_FLOAT64_C(  -705.05) } },
    { { EASYSIMD_FLOAT64_C(   176.42), EASYSIMD_FLOAT64_C(  -570.43) },
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   727.42) },
       INT32_C(         145),
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -570.43) } },
    { { EASYSIMD_FLOAT64_C(  -867.53), EASYSIMD_FLOAT64_C(   506.64) },
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(    89.68) },
       INT32_C(         161),
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   506.64) } },
    { { EASYSIMD_FLOAT64_C(  -965.61), EASYSIMD_FLOAT64_C(   623.64) },
      { EASYSIMD_FLOAT64_C(  -180.71), EASYSIMD_FLOAT64_C(   138.25) },
       INT32_C(         177),
      { EASYSIMD_FLOAT64_C(  -180.71), EASYSIMD_FLOAT64_C(   623.64) } },
    { { EASYSIMD_FLOAT64_C(   320.99), EASYSIMD_FLOAT64_C(   206.55) },
      {       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -760.88) },
       INT32_C(         193),
      {       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   206.55) } },
    { { EASYSIMD_FLOAT64_C(  -464.06), EASYSIMD_FLOAT64_C(  -599.26) },
      { EASYSIMD_FLOAT64_C(   665.68), EASYSIMD_FLOAT64_C(   800.71) },
       INT32_C(         209),
      { EASYSIMD_FLOAT64_C(   665.68), EASYSIMD_FLOAT64_C(  -599.26) } },
    { { EASYSIMD_FLOAT64_C(  -157.90), EASYSIMD_FLOAT64_C(  -769.73) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -430.48) },
       INT32_C(         225),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -769.73) } },
    { { EASYSIMD_FLOAT64_C(   769.01), EASYSIMD_FLOAT64_C(   431.10) },
      { EASYSIMD_FLOAT64_C(  -204.44), EASYSIMD_FLOAT64_C(  -819.06) },
       INT32_C(         241),
      { EASYSIMD_FLOAT64_C(  -204.44), EASYSIMD_FLOAT64_C(   431.10) } },
    { { EASYSIMD_FLOAT64_C(   875.43), EASYSIMD_FLOAT64_C(   665.28) },
      { EASYSIMD_FLOAT64_C(   235.01), EASYSIMD_FLOAT64_C(   909.82) },
       INT32_C(           2),
      { EASYSIMD_FLOAT64_C(   236.00), EASYSIMD_FLOAT64_C(   665.28) } },
    { { EASYSIMD_FLOAT64_C(  -945.70), EASYSIMD_FLOAT64_C(    48.07) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   375.29) },
       INT32_C(          18),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(    48.07) } },
    { { EASYSIMD_FLOAT64_C(   614.41), EASYSIMD_FLOAT64_C(   677.07) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   346.40) },
       INT32_C(          34),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   677.07) } },
    { { EASYSIMD_FLOAT64_C(    12.08), EASYSIMD_FLOAT64_C(  -986.28) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   854.18) },
       INT32_C(          50),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -986.28) } },
    { { EASYSIMD_FLOAT64_C(  -576.30), EASYSIMD_FLOAT64_C(   542.63) },
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -807.29) },
       INT32_C(          66),
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   542.63) } },
    { { EASYSIMD_FLOAT64_C(   494.51), EASYSIMD_FLOAT64_C(   258.93) },
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -270.47) },
       INT32_C(          82),
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   258.93) } },
    { { EASYSIMD_FLOAT64_C(  -783.18), EASYSIMD_FLOAT64_C(   279.46) },
      { EASYSIMD_FLOAT64_C(  -840.88), EASYSIMD_FLOAT64_C(  -528.57) },
       INT32_C(          98),
      { EASYSIMD_FLOAT64_C(  -840.88), EASYSIMD_FLOAT64_C(   279.46) } },
    { { EASYSIMD_FLOAT64_C(   773.53), EASYSIMD_FLOAT64_C(  -851.50) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   119.93) },
       INT32_C(         114),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -851.50) } },
    { { EASYSIMD_FLOAT64_C(  -867.99), EASYSIMD_FLOAT64_C(  -624.77) },
      { EASYSIMD_FLOAT64_C(  -560.77), EASYSIMD_FLOAT64_C(   986.19) },
       INT32_C(         130),
      { EASYSIMD_FLOAT64_C(  -560.77), EASYSIMD_FLOAT64_C(  -624.77) } },
    { { EASYSIMD_FLOAT64_C(   738.25), EASYSIMD_FLOAT64_C(  -590.11) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -673.81) },
       INT32_C(         146),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -590.11) } },
    { { EASYSIMD_FLOAT64_C(   709.69), EASYSIMD_FLOAT64_C(   -23.76) },
      { EASYSIMD_FLOAT64_C(  -369.90), EASYSIMD_FLOAT64_C(   -31.38) },
       INT32_C(         162),
      { EASYSIMD_FLOAT64_C(  -369.90), EASYSIMD_FLOAT64_C(   -23.76) } },
    { { EASYSIMD_FLOAT64_C(   359.63), EASYSIMD_FLOAT64_C(  -862.63) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -856.54) },
       INT32_C(         178),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -862.63) } },
    { { EASYSIMD_FLOAT64_C(  -697.42), EASYSIMD_FLOAT64_C(  -174.38) },
      { EASYSIMD_FLOAT64_C(  -853.10), EASYSIMD_FLOAT64_C(  -923.89) },
       INT32_C(         194),
      { EASYSIMD_FLOAT64_C(  -853.10), EASYSIMD_FLOAT64_C(  -174.38) } },
    { { EASYSIMD_FLOAT64_C(  -977.92), EASYSIMD_FLOAT64_C(   196.04) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   648.75) },
       INT32_C(         210),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   196.04) } },
    { { EASYSIMD_FLOAT64_C(  -912.03), EASYSIMD_FLOAT64_C(   314.24) },
      { EASYSIMD_FLOAT64_C(  -669.90), EASYSIMD_FLOAT64_C(   826.22) },
       INT32_C(         226),
      { EASYSIMD_FLOAT64_C(  -669.90), EASYSIMD_FLOAT64_C(   314.24) } },
    { { EASYSIMD_FLOAT64_C(  -508.05), EASYSIMD_FLOAT64_C(  -847.60) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   627.54) },
       INT32_C(         242),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -847.60) } },
    { { EASYSIMD_FLOAT64_C(  -742.35), EASYSIMD_FLOAT64_C(  -169.29) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   617.27) },
       INT32_C(           3),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -169.29) } },
    { { EASYSIMD_FLOAT64_C(   760.73), EASYSIMD_FLOAT64_C(   322.27) },
      { EASYSIMD_FLOAT64_C(  -716.36), EASYSIMD_FLOAT64_C(  -936.69) },
       INT32_C(          19),
      { EASYSIMD_FLOAT64_C(  -716.00), EASYSIMD_FLOAT64_C(   322.27) } },
    { { EASYSIMD_FLOAT64_C(  -569.46), EASYSIMD_FLOAT64_C(  -860.58) },
      { EASYSIMD_FLOAT64_C(   122.02), EASYSIMD_FLOAT64_C(  -547.38) },
       INT32_C(          35),
      { EASYSIMD_FLOAT64_C(   122.00), EASYSIMD_FLOAT64_C(  -860.58) } },
    { { EASYSIMD_FLOAT64_C(  -542.33), EASYSIMD_FLOAT64_C(  -898.63) },
      {       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -831.46) },
       INT32_C(          51),
      {       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -898.63) } },
    { { EASYSIMD_FLOAT64_C(  -984.44), EASYSIMD_FLOAT64_C(   701.89) },
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -832.04) },
       INT32_C(          67),
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   701.89) } },
    { { EASYSIMD_FLOAT64_C(   331.58), EASYSIMD_FLOAT64_C(  -124.22) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   649.72) },
       INT32_C(          83),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -124.22) } },
    { { EASYSIMD_FLOAT64_C(  -689.12), EASYSIMD_FLOAT64_C(  -746.21) },
      {       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -405.49) },
       INT32_C(          99),
      {       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -746.21) } },
    { { EASYSIMD_FLOAT64_C(  -543.48), EASYSIMD_FLOAT64_C(  -578.95) },
      { EASYSIMD_FLOAT64_C(   477.68), EASYSIMD_FLOAT64_C(   791.98) },
       INT32_C(         115),
      { EASYSIMD_FLOAT64_C(   477.68), EASYSIMD_FLOAT64_C(  -578.95) } },
    { { EASYSIMD_FLOAT64_C(   579.04), EASYSIMD_FLOAT64_C(  -544.50) },
      { EASYSIMD_FLOAT64_C(    47.25), EASYSIMD_FLOAT64_C(   768.38) },
       INT32_C(         131),
      { EASYSIMD_FLOAT64_C(    47.25), EASYSIMD_FLOAT64_C(  -544.50) } },
    { { EASYSIMD_FLOAT64_C(   545.89), EASYSIMD_FLOAT64_C(   783.94) },
      { EASYSIMD_FLOAT64_C(  -864.86), EASYSIMD_FLOAT64_C(  -463.52) },
       INT32_C(         147),
      { EASYSIMD_FLOAT64_C(  -864.86), EASYSIMD_FLOAT64_C(   783.94) } },
    { { EASYSIMD_FLOAT64_C(  -836.24), EASYSIMD_FLOAT64_C(   154.62) },
      { EASYSIMD_FLOAT64_C(   981.95), EASYSIMD_FLOAT64_C(   495.34) },
       INT32_C(         163),
      { EASYSIMD_FLOAT64_C(   981.95), EASYSIMD_FLOAT64_C(   154.62) } },
    { { EASYSIMD_FLOAT64_C(  -157.28), EASYSIMD_FLOAT64_C(   145.06) },
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -328.43) },
       INT32_C(         179),
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   145.06) } },
    { { EASYSIMD_FLOAT64_C(  -949.55), EASYSIMD_FLOAT64_C(    94.34) },
      {       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(    75.50) },
       INT32_C(         195),
      {       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(    94.34) } },
    { { EASYSIMD_FLOAT64_C(   342.83), EASYSIMD_FLOAT64_C(  -578.52) },
      {       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   798.33) },
       INT32_C(         211),
      {       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -578.52) } },
    { { EASYSIMD_FLOAT64_C(    14.63), EASYSIMD_FLOAT64_C(   684.54) },
      { EASYSIMD_FLOAT64_C(  -633.27), EASYSIMD_FLOAT64_C(   551.11) },
       INT32_C(         227),
      { EASYSIMD_FLOAT64_C(  -633.27), EASYSIMD_FLOAT64_C(   684.54) } },
    { { EASYSIMD_FLOAT64_C(  -469.52), EASYSIMD_FLOAT64_C(  -294.27) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -974.18) },
       INT32_C(         243),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -294.27) } },
    { { EASYSIMD_FLOAT64_C(   170.88), EASYSIMD_FLOAT64_C(   259.59) },
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -373.19) },
       INT32_C(           4),
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   259.59) } },
    { { EASYSIMD_FLOAT64_C(   131.17), EASYSIMD_FLOAT64_C(  -922.90) },
      { EASYSIMD_FLOAT64_C(   752.76), EASYSIMD_FLOAT64_C(  -317.98) },
       INT32_C(          20),
      { EASYSIMD_FLOAT64_C(   753.00), EASYSIMD_FLOAT64_C(  -922.90) } },
    { { EASYSIMD_FLOAT64_C(  -694.06), EASYSIMD_FLOAT64_C(  -975.15) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -561.84) },
       INT32_C(          36),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -975.15) } },
    { { EASYSIMD_FLOAT64_C(  -661.24), EASYSIMD_FLOAT64_C(  -945.24) },
      {       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -976.70) },
       INT32_C(          52),
      {       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -945.24) } },
    { { EASYSIMD_FLOAT64_C(   -48.03), EASYSIMD_FLOAT64_C(  -218.45) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   -22.20) },
       INT32_C(          68),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -218.45) } },
    { { EASYSIMD_FLOAT64_C(  -851.32), EASYSIMD_FLOAT64_C(  -222.73) },
      { EASYSIMD_FLOAT64_C(  -128.10), EASYSIMD_FLOAT64_C(  -224.51) },
       INT32_C(          84),
      { EASYSIMD_FLOAT64_C(  -128.09), EASYSIMD_FLOAT64_C(  -222.73) } },
    { { EASYSIMD_FLOAT64_C(   827.27), EASYSIMD_FLOAT64_C(   452.75) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   904.37) },
       INT32_C(         100),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   452.75) } },
    { { EASYSIMD_FLOAT64_C(   524.24), EASYSIMD_FLOAT64_C(   511.46) },
      { EASYSIMD_FLOAT64_C(  -347.86), EASYSIMD_FLOAT64_C(   565.59) },
       INT32_C(         116),
      { EASYSIMD_FLOAT64_C(  -347.86), EASYSIMD_FLOAT64_C(   511.46) } },
    { { EASYSIMD_FLOAT64_C(  -524.67), EASYSIMD_FLOAT64_C(    75.67) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -469.91) },
       INT32_C(         132),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(    75.67) } },
    { { EASYSIMD_FLOAT64_C(   -48.42), EASYSIMD_FLOAT64_C(   676.20) },
      { EASYSIMD_FLOAT64_C(   971.42), EASYSIMD_FLOAT64_C(   903.55) },
       INT32_C(         148),
      { EASYSIMD_FLOAT64_C(   971.42), EASYSIMD_FLOAT64_C(   676.20) } },
    { { EASYSIMD_FLOAT64_C(   249.54), EASYSIMD_FLOAT64_C(  -118.65) },
      { EASYSIMD_FLOAT64_C(   975.42), EASYSIMD_FLOAT64_C(   -11.24) },
       INT32_C(         164),
      { EASYSIMD_FLOAT64_C(   975.42), EASYSIMD_FLOAT64_C(  -118.65) } },
    { { EASYSIMD_FLOAT64_C(  -247.31), EASYSIMD_FLOAT64_C(   860.66) },
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   566.79) },
       INT32_C(         180),
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   860.66) } },
    { { EASYSIMD_FLOAT64_C(   592.31), EASYSIMD_FLOAT64_C(  -536.22) },
      { EASYSIMD_FLOAT64_C(  -860.66), EASYSIMD_FLOAT64_C(   116.54) },
       INT32_C(         196),
      { EASYSIMD_FLOAT64_C(  -860.66), EASYSIMD_FLOAT64_C(  -536.22) } },
    { { EASYSIMD_FLOAT64_C(  -208.51), EASYSIMD_FLOAT64_C(  -317.87) },
      { EASYSIMD_FLOAT64_C(   924.86), EASYSIMD_FLOAT64_C(   266.81) },
       INT32_C(         212),
      { EASYSIMD_FLOAT64_C(   924.86), EASYSIMD_FLOAT64_C(  -317.87) } },
    { { EASYSIMD_FLOAT64_C(  -786.75), EASYSIMD_FLOAT64_C(   796.90) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -475.07) },
       INT32_C(         228),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   796.90) } },
    { { EASYSIMD_FLOAT64_C(  -503.65), EASYSIMD_FLOAT64_C(  -347.96) },
      { EASYSIMD_FLOAT64_C(   492.13), EASYSIMD_FLOAT64_C(   745.89) },
       INT32_C(         244),
      { EASYSIMD_FLOAT64_C(   492.13), EASYSIMD_FLOAT64_C(  -347.96) } },
  };

  easysimd__m128d a, b, r;

  a = easysimd_mm_loadu_pd(test_vec[0].a);
  b = easysimd_mm_loadu_pd(test_vec[0].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(           0));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[1].a);
  b = easysimd_mm_loadu_pd(test_vec[1].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          16));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[2].a);
  b = easysimd_mm_loadu_pd(test_vec[2].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          32));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[3].a);
  b = easysimd_mm_loadu_pd(test_vec[3].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          48));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[4].a);
  b = easysimd_mm_loadu_pd(test_vec[4].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          64));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[5].a);
  b = easysimd_mm_loadu_pd(test_vec[5].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          80));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[6].a);
  b = easysimd_mm_loadu_pd(test_vec[6].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          96));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[7].a);
  b = easysimd_mm_loadu_pd(test_vec[7].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         112));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[7].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[8].a);
  b = easysimd_mm_loadu_pd(test_vec[8].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         128));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[8].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[9].a);
  b = easysimd_mm_loadu_pd(test_vec[9].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         144));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[9].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[10].a);
  b = easysimd_mm_loadu_pd(test_vec[10].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         160));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[10].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[11].a);
  b = easysimd_mm_loadu_pd(test_vec[11].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         176));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[11].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[12].a);
  b = easysimd_mm_loadu_pd(test_vec[12].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         192));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[12].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[13].a);
  b = easysimd_mm_loadu_pd(test_vec[13].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         208));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[13].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[14].a);
  b = easysimd_mm_loadu_pd(test_vec[14].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         224));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[14].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[15].a);
  b = easysimd_mm_loadu_pd(test_vec[15].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         240));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[15].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[16].a);
  b = easysimd_mm_loadu_pd(test_vec[16].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(           1));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[16].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[17].a);
  b = easysimd_mm_loadu_pd(test_vec[17].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          17));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[17].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[18].a);
  b = easysimd_mm_loadu_pd(test_vec[18].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          33));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[18].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[19].a);
  b = easysimd_mm_loadu_pd(test_vec[19].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          49));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[19].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[20].a);
  b = easysimd_mm_loadu_pd(test_vec[20].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          65));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[20].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[21].a);
  b = easysimd_mm_loadu_pd(test_vec[21].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          81));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[21].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[22].a);
  b = easysimd_mm_loadu_pd(test_vec[22].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          97));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[22].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[23].a);
  b = easysimd_mm_loadu_pd(test_vec[23].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         113));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[23].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[24].a);
  b = easysimd_mm_loadu_pd(test_vec[24].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         129));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[24].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[25].a);
  b = easysimd_mm_loadu_pd(test_vec[25].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         145));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[25].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[26].a);
  b = easysimd_mm_loadu_pd(test_vec[26].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         161));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[26].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[27].a);
  b = easysimd_mm_loadu_pd(test_vec[27].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         177));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[27].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[28].a);
  b = easysimd_mm_loadu_pd(test_vec[28].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         193));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[28].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[29].a);
  b = easysimd_mm_loadu_pd(test_vec[29].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         209));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[29].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[30].a);
  b = easysimd_mm_loadu_pd(test_vec[30].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         225));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[30].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[31].a);
  b = easysimd_mm_loadu_pd(test_vec[31].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         241));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[31].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[32].a);
  b = easysimd_mm_loadu_pd(test_vec[32].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(           2));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[32].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[33].a);
  b = easysimd_mm_loadu_pd(test_vec[33].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          18));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[33].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[34].a);
  b = easysimd_mm_loadu_pd(test_vec[34].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          34));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[34].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[35].a);
  b = easysimd_mm_loadu_pd(test_vec[35].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          50));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[35].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[36].a);
  b = easysimd_mm_loadu_pd(test_vec[36].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          66));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[36].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[37].a);
  b = easysimd_mm_loadu_pd(test_vec[37].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          82));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[37].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[38].a);
  b = easysimd_mm_loadu_pd(test_vec[38].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          98));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[38].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[39].a);
  b = easysimd_mm_loadu_pd(test_vec[39].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         114));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[39].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[40].a);
  b = easysimd_mm_loadu_pd(test_vec[40].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         130));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[40].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[41].a);
  b = easysimd_mm_loadu_pd(test_vec[41].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         146));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[41].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[42].a);
  b = easysimd_mm_loadu_pd(test_vec[42].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         162));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[42].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[43].a);
  b = easysimd_mm_loadu_pd(test_vec[43].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         178));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[43].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[44].a);
  b = easysimd_mm_loadu_pd(test_vec[44].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         194));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[44].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[45].a);
  b = easysimd_mm_loadu_pd(test_vec[45].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         210));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[45].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[46].a);
  b = easysimd_mm_loadu_pd(test_vec[46].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         226));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[46].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[47].a);
  b = easysimd_mm_loadu_pd(test_vec[47].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         242));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[47].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[48].a);
  b = easysimd_mm_loadu_pd(test_vec[48].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(           3));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[48].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[49].a);
  b = easysimd_mm_loadu_pd(test_vec[49].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          19));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[49].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[50].a);
  b = easysimd_mm_loadu_pd(test_vec[50].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          35));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[50].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[51].a);
  b = easysimd_mm_loadu_pd(test_vec[51].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          51));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[51].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[52].a);
  b = easysimd_mm_loadu_pd(test_vec[52].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          67));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[52].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[53].a);
  b = easysimd_mm_loadu_pd(test_vec[53].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          83));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[53].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[54].a);
  b = easysimd_mm_loadu_pd(test_vec[54].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          99));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[54].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[55].a);
  b = easysimd_mm_loadu_pd(test_vec[55].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         115));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[55].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[56].a);
  b = easysimd_mm_loadu_pd(test_vec[56].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         131));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[56].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[57].a);
  b = easysimd_mm_loadu_pd(test_vec[57].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         147));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[57].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[58].a);
  b = easysimd_mm_loadu_pd(test_vec[58].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         163));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[58].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[59].a);
  b = easysimd_mm_loadu_pd(test_vec[59].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         179));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[59].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[60].a);
  b = easysimd_mm_loadu_pd(test_vec[60].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         195));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[60].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[61].a);
  b = easysimd_mm_loadu_pd(test_vec[61].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         211));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[61].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[62].a);
  b = easysimd_mm_loadu_pd(test_vec[62].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         227));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[62].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[63].a);
  b = easysimd_mm_loadu_pd(test_vec[63].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         243));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[63].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[64].a);
  b = easysimd_mm_loadu_pd(test_vec[64].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(           4));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[64].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[65].a);
  b = easysimd_mm_loadu_pd(test_vec[65].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          20));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[65].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[66].a);
  b = easysimd_mm_loadu_pd(test_vec[66].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          36));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[66].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[67].a);
  b = easysimd_mm_loadu_pd(test_vec[67].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          52));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[67].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[68].a);
  b = easysimd_mm_loadu_pd(test_vec[68].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          68));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[68].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[69].a);
  b = easysimd_mm_loadu_pd(test_vec[69].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(          84));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[69].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[70].a);
  b = easysimd_mm_loadu_pd(test_vec[70].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         100));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[70].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[71].a);
  b = easysimd_mm_loadu_pd(test_vec[71].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         116));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[71].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[72].a);
  b = easysimd_mm_loadu_pd(test_vec[72].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         132));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[72].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[73].a);
  b = easysimd_mm_loadu_pd(test_vec[73].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         148));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[73].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[74].a);
  b = easysimd_mm_loadu_pd(test_vec[74].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         164));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[74].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[75].a);
  b = easysimd_mm_loadu_pd(test_vec[75].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         180));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[75].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[76].a);
  b = easysimd_mm_loadu_pd(test_vec[76].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         196));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[76].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[77].a);
  b = easysimd_mm_loadu_pd(test_vec[77].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         212));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[77].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[78].a);
  b = easysimd_mm_loadu_pd(test_vec[78].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         228));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[78].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[79].a);
  b = easysimd_mm_loadu_pd(test_vec[79].b);
  r = easysimd_mm_roundscale_sd(a, b, INT32_C(         244));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[79].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 5 ; i++) {
    for (int j = 0 ; j < 16 ; j++) {
      easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
      easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
      if((easysimd_test_codegen_rand() & 1)) {
        if((easysimd_test_codegen_rand() & 1))
          b = easysimd_mm_mask_mov_pd(b, 1, easysimd_mm_set1_pd(EASYSIMD_MATH_NAN));
        else {
          if((easysimd_test_codegen_rand() & 1))
            b = easysimd_mm_mask_mov_pd(b, 1, easysimd_mm_set1_pd(EASYSIMD_MATH_INFINITY));
          else
            b = easysimd_mm_mask_mov_pd(b, 1, easysimd_mm_set1_pd(-EASYSIMD_MATH_INFINITY));
        }
      }
      int imm8 = ((j << 4) | round_type[i]) & 255;
      easysimd__m128d r;
      EASYSIMD_CONSTIFY_256_(easysimd_mm_roundscale_sd, r, easysimd_mm_setzero_pd(), imm8, a, b);

      easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
      easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
      easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
      easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
    }
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_roundscale_sd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[2];
    const easysimd__mmask8 k;
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const int32_t imm8;
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -152.89), EASYSIMD_FLOAT64_C(   255.35) },
      UINT8_C(242),
      { EASYSIMD_FLOAT64_C(   575.09), EASYSIMD_FLOAT64_C(   308.20) },
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -284.87) },
       INT32_C(         224),
      { EASYSIMD_FLOAT64_C(  -152.89), EASYSIMD_FLOAT64_C(   308.20) } },
    { { EASYSIMD_FLOAT64_C(  -592.52), EASYSIMD_FLOAT64_C(   699.67) },
      UINT8_C(134),
      { EASYSIMD_FLOAT64_C(   489.75), EASYSIMD_FLOAT64_C(   904.42) },
      { EASYSIMD_FLOAT64_C(  -814.03), EASYSIMD_FLOAT64_C(  -202.13) },
       INT32_C(         177),
      { EASYSIMD_FLOAT64_C(  -592.52), EASYSIMD_FLOAT64_C(   904.42) } },
    { { EASYSIMD_FLOAT64_C(   531.17), EASYSIMD_FLOAT64_C(   420.57) },
      UINT8_C(140),
      { EASYSIMD_FLOAT64_C(  -721.46), EASYSIMD_FLOAT64_C(   -85.09) },
      { EASYSIMD_FLOAT64_C(  -540.17), EASYSIMD_FLOAT64_C(   877.77) },
       INT32_C(          98),
      { EASYSIMD_FLOAT64_C(   531.17), EASYSIMD_FLOAT64_C(   -85.09) } },
    { { EASYSIMD_FLOAT64_C(  -178.78), EASYSIMD_FLOAT64_C(  -745.00) },
      UINT8_C( 67),
      { EASYSIMD_FLOAT64_C(  -923.43), EASYSIMD_FLOAT64_C(   247.60) },
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   384.78) },
       INT32_C(         179),
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   247.60) } },
    { { EASYSIMD_FLOAT64_C(   840.86), EASYSIMD_FLOAT64_C(  -204.15) },
      UINT8_C( 77),
      { EASYSIMD_FLOAT64_C(   540.53), EASYSIMD_FLOAT64_C(   -50.72) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   444.96) },
       INT32_C(         132),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   -50.72) } },
    { { EASYSIMD_FLOAT64_C(    49.85), EASYSIMD_FLOAT64_C(   550.73) },
      UINT8_C( 67),
      { EASYSIMD_FLOAT64_C(  -882.69), EASYSIMD_FLOAT64_C(   829.28) },
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -422.86) },
       INT32_C(         176),
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   829.28) } },
    { { EASYSIMD_FLOAT64_C(   530.79), EASYSIMD_FLOAT64_C(   309.38) },
      UINT8_C( 29),
      { EASYSIMD_FLOAT64_C(  -221.61), EASYSIMD_FLOAT64_C(  -325.85) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   843.82) },
       INT32_C(         225),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -325.85) } },
    { { EASYSIMD_FLOAT64_C(  -405.09), EASYSIMD_FLOAT64_C(  -883.54) },
      UINT8_C(139),
      { EASYSIMD_FLOAT64_C(  -864.56), EASYSIMD_FLOAT64_C(    65.73) },
      { EASYSIMD_FLOAT64_C(  -878.07), EASYSIMD_FLOAT64_C(   580.40) },
       INT32_C(          82),
      { EASYSIMD_FLOAT64_C(  -878.06), EASYSIMD_FLOAT64_C(    65.73) } },
  };

  easysimd__m128d src, a, b, r;

  src = easysimd_mm_loadu_pd(test_vec[0].src);
  a = easysimd_mm_loadu_pd(test_vec[0].a);
  b = easysimd_mm_loadu_pd(test_vec[0].b);
  r = easysimd_mm_mask_roundscale_sd(src, test_vec[0].k, a, b, INT32_C(         224));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[0].r), 1);

  src = easysimd_mm_loadu_pd(test_vec[1].src);
  a = easysimd_mm_loadu_pd(test_vec[1].a);
  b = easysimd_mm_loadu_pd(test_vec[1].b);
  r = easysimd_mm_mask_roundscale_sd(src, test_vec[1].k, a, b, INT32_C(         177));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[1].r), 1);

  src = easysimd_mm_loadu_pd(test_vec[2].src);
  a = easysimd_mm_loadu_pd(test_vec[2].a);
  b = easysimd_mm_loadu_pd(test_vec[2].b);
  r = easysimd_mm_mask_roundscale_sd(src, test_vec[2].k, a, b, INT32_C(          98));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[2].r), 1);

  src = easysimd_mm_loadu_pd(test_vec[3].src);
  a = easysimd_mm_loadu_pd(test_vec[3].a);
  b = easysimd_mm_loadu_pd(test_vec[3].b);
  r = easysimd_mm_mask_roundscale_sd(src, test_vec[3].k, a, b, INT32_C(         179));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[3].r), 1);

  src = easysimd_mm_loadu_pd(test_vec[4].src);
  a = easysimd_mm_loadu_pd(test_vec[4].a);
  b = easysimd_mm_loadu_pd(test_vec[4].b);
  r = easysimd_mm_mask_roundscale_sd(src, test_vec[4].k, a, b, INT32_C(         132));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[4].r), 1);

  src = easysimd_mm_loadu_pd(test_vec[5].src);
  a = easysimd_mm_loadu_pd(test_vec[5].a);
  b = easysimd_mm_loadu_pd(test_vec[5].b);
  r = easysimd_mm_mask_roundscale_sd(src, test_vec[5].k, a, b, INT32_C(         176));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[5].r), 1);

  src = easysimd_mm_loadu_pd(test_vec[6].src);
  a = easysimd_mm_loadu_pd(test_vec[6].a);
  b = easysimd_mm_loadu_pd(test_vec[6].b);
  r = easysimd_mm_mask_roundscale_sd(src, test_vec[6].k, a, b, INT32_C(         225));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[6].r), 1);

  src = easysimd_mm_loadu_pd(test_vec[7].src);
  a = easysimd_mm_loadu_pd(test_vec[7].a);
  b = easysimd_mm_loadu_pd(test_vec[7].b);
  r = easysimd_mm_mask_roundscale_sd(src, test_vec[7].k, a, b, INT32_C(          82));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d src = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    if((easysimd_test_codegen_rand() & 1)) {
      if((easysimd_test_codegen_rand() & 1))
        b = easysimd_mm_mask_mov_pd(b, 1, easysimd_mm_set1_pd(EASYSIMD_MATH_NAN));
      else {
        if((easysimd_test_codegen_rand() & 1))
          b = easysimd_mm_mask_mov_pd(b, 1, easysimd_mm_set1_pd(EASYSIMD_MATH_INFINITY));
        else
          b = easysimd_mm_mask_mov_pd(b, 1, easysimd_mm_set1_pd(-EASYSIMD_MATH_INFINITY));
      }
    }
    int imm8 = (((easysimd_test_codegen_rand() & 15) << 4) | round_type[i % 5]) & 255;
    easysimd__m128d r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm_mask_roundscale_sd, r, easysimd_mm_setzero_pd(), imm8, src, k, a, b);

    easysimd_test_x86_write_f64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_roundscale_sd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const int32_t imm8;
    const easysimd_float64 r[2];
  } test_vec[] = {
    { UINT8_C(231),
      { EASYSIMD_FLOAT64_C(   679.03), EASYSIMD_FLOAT64_C(   330.13) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   170.83) },
       INT32_C(          48),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   330.13) } },
    { UINT8_C(227),
      { EASYSIMD_FLOAT64_C(  -640.76), EASYSIMD_FLOAT64_C(  -486.00) },
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -204.74) },
       INT32_C(         209),
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -486.00) } },
    { UINT8_C(219),
      { EASYSIMD_FLOAT64_C(   264.61), EASYSIMD_FLOAT64_C(  -297.73) },
      { EASYSIMD_FLOAT64_C(   370.03), EASYSIMD_FLOAT64_C(   192.79) },
       INT32_C(         162),
      { EASYSIMD_FLOAT64_C(   370.03), EASYSIMD_FLOAT64_C(  -297.73) } },
    { UINT8_C(225),
      { EASYSIMD_FLOAT64_C(  -946.41), EASYSIMD_FLOAT64_C(  -365.05) },
      { EASYSIMD_FLOAT64_C(  -822.97), EASYSIMD_FLOAT64_C(   822.58) },
       INT32_C(           3),
      { EASYSIMD_FLOAT64_C(  -822.00), EASYSIMD_FLOAT64_C(  -365.05) } },
    { UINT8_C( 14),
      { EASYSIMD_FLOAT64_C(   101.90), EASYSIMD_FLOAT64_C(  -335.55) },
      {       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -727.27) },
       INT32_C(          68),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -335.55) } },
    { UINT8_C(201),
      { EASYSIMD_FLOAT64_C(   337.11), EASYSIMD_FLOAT64_C(  -368.49) },
      { EASYSIMD_FLOAT64_C(   497.71), EASYSIMD_FLOAT64_C(  -156.35) },
       INT32_C(          32),
      { EASYSIMD_FLOAT64_C(   497.75), EASYSIMD_FLOAT64_C(  -368.49) } },
    { UINT8_C(128),
      { EASYSIMD_FLOAT64_C(   229.27), EASYSIMD_FLOAT64_C(   805.61) },
      { EASYSIMD_FLOAT64_C(   385.54), EASYSIMD_FLOAT64_C(  -400.70) },
       INT32_C(          97),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   805.61) } },
    { UINT8_C(193),
      { EASYSIMD_FLOAT64_C(   -90.84), EASYSIMD_FLOAT64_C(  -911.13) },
      { EASYSIMD_FLOAT64_C(   474.67), EASYSIMD_FLOAT64_C(    86.19) },
       INT32_C(           2),
      { EASYSIMD_FLOAT64_C(   475.00), EASYSIMD_FLOAT64_C(  -911.13) } },
  };

  easysimd__m128d a, b, r;

  a = easysimd_mm_loadu_pd(test_vec[0].a);
  b = easysimd_mm_loadu_pd(test_vec[0].b);
  r = easysimd_mm_maskz_roundscale_sd(test_vec[0].k, a, b, INT32_C(          48));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[1].a);
  b = easysimd_mm_loadu_pd(test_vec[1].b);
  r = easysimd_mm_maskz_roundscale_sd(test_vec[1].k, a, b, INT32_C(         209));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[2].a);
  b = easysimd_mm_loadu_pd(test_vec[2].b);
  r = easysimd_mm_maskz_roundscale_sd(test_vec[2].k, a, b, INT32_C(         162));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[3].a);
  b = easysimd_mm_loadu_pd(test_vec[3].b);
  r = easysimd_mm_maskz_roundscale_sd(test_vec[3].k, a, b, INT32_C(           3));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[4].a);
  b = easysimd_mm_loadu_pd(test_vec[4].b);
  r = easysimd_mm_maskz_roundscale_sd(test_vec[4].k, a, b, INT32_C(          68));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[5].a);
  b = easysimd_mm_loadu_pd(test_vec[5].b);
  r = easysimd_mm_maskz_roundscale_sd(test_vec[5].k, a, b, INT32_C(          32));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[6].a);
  b = easysimd_mm_loadu_pd(test_vec[6].b);
  r = easysimd_mm_maskz_roundscale_sd(test_vec[6].k, a, b, INT32_C(          97));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[7].a);
  b = easysimd_mm_loadu_pd(test_vec[7].b);
  r = easysimd_mm_maskz_roundscale_sd(test_vec[7].k, a, b, INT32_C(           2));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  int round_type[5] = {EASYSIMD_MM_FROUND_TO_NEAREST_INT, EASYSIMD_MM_FROUND_TO_NEG_INF, EASYSIMD_MM_FROUND_TO_POS_INF, EASYSIMD_MM_FROUND_TO_ZERO, EASYSIMD_MM_FROUND_CUR_DIRECTION};
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    if((easysimd_test_codegen_rand() & 1)) {
      if((easysimd_test_codegen_rand() & 1))
        b = easysimd_mm_mask_mov_pd(b, 1, easysimd_mm_set1_pd(EASYSIMD_MATH_NAN));
      else {
        if((easysimd_test_codegen_rand() & 1))
          b = easysimd_mm_mask_mov_pd(b, 1, easysimd_mm_set1_pd(EASYSIMD_MATH_INFINITY));
        else
          b = easysimd_mm_mask_mov_pd(b, 1, easysimd_mm_set1_pd(-EASYSIMD_MATH_INFINITY));
      }
    }
    int imm8 = (((easysimd_test_codegen_rand() & 15) << 4) | round_type[i % 5]) & 255;
    easysimd__m128d r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm_maskz_roundscale_sd, r, easysimd_mm_setzero_pd(), imm8, k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_roundscale_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_roundscale_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_roundscale_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_roundscale_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_roundscale_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_roundscale_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_roundscale_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_roundscale_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_roundscale_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_roundscale_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_roundscale_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_roundscale_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_roundscale_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_roundscale_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_roundscale_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_roundscale_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_roundscale_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_roundscale_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_roundscale_ss)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_roundscale_ss)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_roundscale_ss)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_roundscale_sd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_roundscale_sd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_roundscale_sd)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
