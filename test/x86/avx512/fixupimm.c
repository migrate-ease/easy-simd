#define EASYSIMD_TEST_X86_AVX512_INSN fixupimm

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/fixupimm.h>
#include <easysimd/x86/avx512/setzero.h>

static int
test_easysimd_mm_fixupimm_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const int32_t c[4];
    const int imm8;
    const easysimd_float32 r[4];
  } test_vec[] = {
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   201.54),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   881.13) },
      { EASYSIMD_FLOAT32_C(   -46.40),            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   174.98) },
      { -INT32_C(   408511845),  INT32_C(  1587971559), -INT32_C(   627447801),  INT32_C(  2001924042) },
       INT32_C(           9),
      { EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    -0.00) } },
    { { EASYSIMD_FLOAT32_C(   664.12), EASYSIMD_FLOAT32_C(  -562.89), EASYSIMD_FLOAT32_C(   663.33), EASYSIMD_FLOAT32_C(   970.48) },
      { EASYSIMD_FLOAT32_C(   603.80), EASYSIMD_FLOAT32_C(  -265.96), EASYSIMD_FLOAT32_C(  -984.77), EASYSIMD_FLOAT32_C(   917.02) },
      { -INT32_C(   718677047),  INT32_C(   781642408),  INT32_C(    34931302), -INT32_C(  2081834084) },
       INT32_C(          65),
      { EASYSIMD_FLOAT32_C(     1.57), EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     0.00) } },
    { { EASYSIMD_FLOAT32_C(   517.09), EASYSIMD_FLOAT32_C(  -513.21), EASYSIMD_FLOAT32_C(  -326.22), EASYSIMD_FLOAT32_C(  -832.74) },
      { EASYSIMD_FLOAT32_C(  -388.67), EASYSIMD_FLOAT32_C(   -81.18), EASYSIMD_FLOAT32_C(   227.41), EASYSIMD_FLOAT32_C(  -456.50) },
      {  INT32_C(  1126398840),  INT32_C(  1941599594),  INT32_C(  1252359041), -INT32_C(    14627242) },
       INT32_C(         184),
      {            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF,     -EASYSIMD_MATH_INFINITYF,  EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00) } },
    { { EASYSIMD_FLOAT32_C(  -446.07), EASYSIMD_FLOAT32_C(  -715.24), EASYSIMD_FLOAT32_C(  -742.45), EASYSIMD_FLOAT32_C(  -938.47) },
      { EASYSIMD_FLOAT32_C(   179.77), EASYSIMD_FLOAT32_C(   -73.95), EASYSIMD_FLOAT32_C(    38.40), EASYSIMD_FLOAT32_C(   117.23) },
      {  INT32_C(  1428177592),  INT32_C(  1071123198),  INT32_C(   310885018), -INT32_C(  1621775819) },
       INT32_C(          32),
      {      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(    38.40), EASYSIMD_FLOAT32_C(    -1.00) } },
    { { EASYSIMD_FLOAT32_C(  -872.41), EASYSIMD_FLOAT32_C(   469.38), EASYSIMD_FLOAT32_C(    -1.64), EASYSIMD_FLOAT32_C(    81.20) },
      { EASYSIMD_FLOAT32_C(   969.80), EASYSIMD_FLOAT32_C(   475.02), EASYSIMD_FLOAT32_C(  -743.82), EASYSIMD_FLOAT32_C(   633.92) },
      { -INT32_C(   504580214),  INT32_C(  1038093445),  INT32_C(  2069564866),  INT32_C(  1322286160) },
       INT32_C(         134),
      { EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     0.50),     -EASYSIMD_MATH_INFINITYF } },
    { { EASYSIMD_FLOAT32_C(   912.13), EASYSIMD_FLOAT32_C(   919.50), EASYSIMD_FLOAT32_C(   604.40), EASYSIMD_FLOAT32_C(   515.94) },
      { EASYSIMD_FLOAT32_C(  -346.46), EASYSIMD_FLOAT32_C(   619.63), EASYSIMD_FLOAT32_C(   432.96), EASYSIMD_FLOAT32_C(  -829.36) },
      { -INT32_C(  1758325662), -INT32_C(   550074177),  INT32_C(   578832535), -INT32_C(  2080150273) },
       INT32_C(         120),
      { EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.57),            EASYSIMD_MATH_NANF,     -EASYSIMD_MATH_INFINITYF } },
    { { EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00) },
      { -INT32_C(  1289243822), -INT32_C(  1566703114), -INT32_C(  1814574289), -INT32_C(   993823156) },
       INT32_C(           6),
      { EASYSIMD_FLOAT32_C(     0.50), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    90.00),      EASYSIMD_MATH_INFINITYF } },
    { { EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) },
      { EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00) },
      {  INT32_C(   442678001), -INT32_C(  1898682214), -INT32_C(  1314041761), -INT32_C(   110767101) },
       INT32_C(         213),
      { EASYSIMD_FLOAT32_C(     0.50), EASYSIMD_FLOAT32_C(     0.00),      EASYSIMD_MATH_INFINITYF,     -EASYSIMD_MATH_INFINITYF } },
  };

  easysimd__m128 a, b, r;
  easysimd__m128i c;

  a = easysimd_mm_loadu_ps(test_vec[0].a);
  b = easysimd_mm_loadu_ps(test_vec[0].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[0].c);
  r = easysimd_mm_fixupimm_ps(a, b, c, INT32_C(           9));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[1].a);
  b = easysimd_mm_loadu_ps(test_vec[1].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[1].c);
  r = easysimd_mm_fixupimm_ps(a, b, c, INT32_C(          65));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[2].a);
  b = easysimd_mm_loadu_ps(test_vec[2].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[2].c);
  r = easysimd_mm_fixupimm_ps(a, b, c, INT32_C(         184));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[3].a);
  b = easysimd_mm_loadu_ps(test_vec[3].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[3].c);
  r = easysimd_mm_fixupimm_ps(a, b, c, INT32_C(          32));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[4].a);
  b = easysimd_mm_loadu_ps(test_vec[4].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[4].c);
  r = easysimd_mm_fixupimm_ps(a, b, c, INT32_C(         134));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[5].a);
  b = easysimd_mm_loadu_ps(test_vec[5].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[5].c);
  r = easysimd_mm_fixupimm_ps(a, b, c, INT32_C(         120));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[6].a);
  b = easysimd_mm_loadu_ps(test_vec[6].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[6].c);
  r = easysimd_mm_fixupimm_ps(a, b, c, INT32_C(           6));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[7].a);
  b = easysimd_mm_loadu_ps(test_vec[7].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[7].c);
  r = easysimd_mm_fixupimm_ps(a, b, c, INT32_C(         213));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float32 values[8 * 2 * sizeof(easysimd__m128)];
  easysimd_test_x86_random_f32x4_full(8, 2, values, -1000.0f, 1000.0f, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m128 a = easysimd_test_x86_random_extract_f32x4(i, 2, 0, values);
    easysimd__m128 b = easysimd_test_x86_random_extract_f32x4(i, 2, 1, values);
    easysimd__m128i c = easysimd_test_x86_random_i32x4();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m128 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm_fixupimm_ps, r, easysimd_mm_setzero_ps(), imm8, a, b, c);

    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_fixupimm_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const easysimd__mmask8 k;
    const easysimd_float32 b[4];
    const int32_t c[4];
    const int imm8;
    const easysimd_float32 r[4];
  } test_vec[] = {
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    23.48),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   523.55) },
      UINT8_C( 56),
      { EASYSIMD_FLOAT32_C(   814.83),            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -987.74) },
      { -INT32_C(  1062186756), -INT32_C(    13150401),  INT32_C(   518961091), -INT32_C(   970966419) },
       INT32_C(         106),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    23.48),            EASYSIMD_MATH_NANF,     -EASYSIMD_MATH_INFINITYF } },
    { { EASYSIMD_FLOAT32_C(   167.41), EASYSIMD_FLOAT32_C(  -298.70), EASYSIMD_FLOAT32_C(    73.34), EASYSIMD_FLOAT32_C(  -860.50) },
      UINT8_C(168),
      { EASYSIMD_FLOAT32_C(  -151.65), EASYSIMD_FLOAT32_C(  -558.71), EASYSIMD_FLOAT32_C(  -472.09), EASYSIMD_FLOAT32_C(  -613.35) },
      {  INT32_C(   597534342),  INT32_C(  1360845833),  INT32_C(    72177076),  INT32_C(  1564675589) },
       INT32_C(          69),
      { EASYSIMD_FLOAT32_C(   167.41), EASYSIMD_FLOAT32_C(  -298.70), EASYSIMD_FLOAT32_C(    73.34), EASYSIMD_FLOAT32_C(     1.57) } },
    { { EASYSIMD_FLOAT32_C(  -954.81), EASYSIMD_FLOAT32_C(  -400.16), EASYSIMD_FLOAT32_C(  -515.11), EASYSIMD_FLOAT32_C(   674.37) },
      UINT8_C( 48),
      { EASYSIMD_FLOAT32_C(  -640.90), EASYSIMD_FLOAT32_C(  -783.28), EASYSIMD_FLOAT32_C(   990.54), EASYSIMD_FLOAT32_C(   -77.93) },
      {  INT32_C(  1584032062), -INT32_C(  1735470033), -INT32_C(   333564346), -INT32_C(  1342881325) },
       INT32_C(          94),
      { EASYSIMD_FLOAT32_C(  -954.81), EASYSIMD_FLOAT32_C(  -400.16), EASYSIMD_FLOAT32_C(  -515.11), EASYSIMD_FLOAT32_C(   674.37) } },
    { { EASYSIMD_FLOAT32_C(  -839.98), EASYSIMD_FLOAT32_C(   384.67), EASYSIMD_FLOAT32_C(  -872.97), EASYSIMD_FLOAT32_C(   560.86) },
      UINT8_C(147),
      { EASYSIMD_FLOAT32_C(   200.25), EASYSIMD_FLOAT32_C(   749.68), EASYSIMD_FLOAT32_C(   -70.41), EASYSIMD_FLOAT32_C(  -172.70) },
      { -INT32_C(  1432242073),  INT32_C(   904783381),  INT32_C(  1265835490),  INT32_C(  1551618696) },
       INT32_C(          96),
      { EASYSIMD_FLOAT32_C(     1.00),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -872.97), EASYSIMD_FLOAT32_C(   560.86) } },
    { { EASYSIMD_FLOAT32_C(  -226.84), EASYSIMD_FLOAT32_C(   416.74), EASYSIMD_FLOAT32_C(  -649.15), EASYSIMD_FLOAT32_C(  -412.01) },
      UINT8_C( 49),
      { EASYSIMD_FLOAT32_C(   623.67), EASYSIMD_FLOAT32_C(   586.37), EASYSIMD_FLOAT32_C(  -399.76), EASYSIMD_FLOAT32_C(  -208.92) },
      { -INT32_C(  2089653874),  INT32_C(  2093600792),  INT32_C(  1021533571), -INT32_C(   447639810) },
       INT32_C(         123),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   416.74), EASYSIMD_FLOAT32_C(  -649.15), EASYSIMD_FLOAT32_C(  -412.01) } },
    { { EASYSIMD_FLOAT32_C(  -712.33), EASYSIMD_FLOAT32_C(   673.58), EASYSIMD_FLOAT32_C(   -69.41), EASYSIMD_FLOAT32_C(   136.02) },
      UINT8_C(250),
      { EASYSIMD_FLOAT32_C(  -885.13), EASYSIMD_FLOAT32_C(   458.50), EASYSIMD_FLOAT32_C(   522.67), EASYSIMD_FLOAT32_C(  -839.94) },
      { -INT32_C(  1899225069),  INT32_C(   530656381),  INT32_C(   732877506),  INT32_C(   356790596) },
       INT32_C(         251),
      { EASYSIMD_FLOAT32_C(  -712.33), EASYSIMD_FLOAT32_C(   458.50), EASYSIMD_FLOAT32_C(   -69.41),      EASYSIMD_MATH_INFINITYF } },
    { { EASYSIMD_FLOAT32_C(  -941.67), EASYSIMD_FLOAT32_C(  -992.44), EASYSIMD_FLOAT32_C(   834.43), EASYSIMD_FLOAT32_C(  -582.57) },
      UINT8_C(163),
      { EASYSIMD_FLOAT32_C(  -775.72), EASYSIMD_FLOAT32_C(   824.97), EASYSIMD_FLOAT32_C(   339.51), EASYSIMD_FLOAT32_C(  -615.70) },
      {  INT32_C(   640767700),  INT32_C(    61713467),  INT32_C(  1695983429), -INT32_C(  1595759500) },
       INT32_C(          69),
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -992.44), EASYSIMD_FLOAT32_C(   834.43), EASYSIMD_FLOAT32_C(  -582.57) } },
    { { EASYSIMD_FLOAT32_C(   209.64), EASYSIMD_FLOAT32_C(   466.53), EASYSIMD_FLOAT32_C(   945.16), EASYSIMD_FLOAT32_C(  -590.11) },
      UINT8_C(176),
      { EASYSIMD_FLOAT32_C(   216.22), EASYSIMD_FLOAT32_C(  -125.25), EASYSIMD_FLOAT32_C(   237.19), EASYSIMD_FLOAT32_C(   989.38) },
      { -INT32_C(   756982898),  INT32_C(   160619632), -INT32_C(  1948436940),  INT32_C(   348521319) },
       INT32_C(         176),
      { EASYSIMD_FLOAT32_C(   209.64), EASYSIMD_FLOAT32_C(   466.53), EASYSIMD_FLOAT32_C(   945.16), EASYSIMD_FLOAT32_C(  -590.11) } },
  };

  easysimd__m128 a, b, r;
  easysimd__m128i c;

  a = easysimd_mm_loadu_ps(test_vec[0].a);
  b = easysimd_mm_loadu_ps(test_vec[0].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[0].c);
  r = easysimd_mm_mask_fixupimm_ps(a, test_vec[0].k, b, c, INT32_C(         106));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[1].a);
  b = easysimd_mm_loadu_ps(test_vec[1].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[1].c);
  r = easysimd_mm_mask_fixupimm_ps(a, test_vec[1].k, b, c, INT32_C(          69));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[2].a);
  b = easysimd_mm_loadu_ps(test_vec[2].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[2].c);
  r = easysimd_mm_mask_fixupimm_ps(a, test_vec[2].k, b, c, INT32_C(          94));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[3].a);
  b = easysimd_mm_loadu_ps(test_vec[3].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[3].c);
  r = easysimd_mm_mask_fixupimm_ps(a, test_vec[3].k, b, c, INT32_C(          96));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[4].a);
  b = easysimd_mm_loadu_ps(test_vec[4].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[4].c);
  r = easysimd_mm_mask_fixupimm_ps(a, test_vec[4].k, b, c, INT32_C(         123));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[5].a);
  b = easysimd_mm_loadu_ps(test_vec[5].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[5].c);
  r = easysimd_mm_mask_fixupimm_ps(a, test_vec[5].k, b, c, INT32_C(         251));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[6].a);
  b = easysimd_mm_loadu_ps(test_vec[6].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[6].c);
  r = easysimd_mm_mask_fixupimm_ps(a, test_vec[6].k, b, c, INT32_C(          69));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[7].a);
  b = easysimd_mm_loadu_ps(test_vec[7].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[7].c);
  r = easysimd_mm_mask_fixupimm_ps(a, test_vec[7].k, b, c, INT32_C(         176));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float32 values[8 * 2 * sizeof(easysimd__m128)];
  easysimd_test_x86_random_f32x4_full(8, 2, values, -1000.0f, 1000.0f, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m128 a = easysimd_test_x86_random_extract_f32x4(i, 2, 0, values);
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 b = easysimd_test_x86_random_extract_f32x4(i, 2, 1, values);
    easysimd__m128i c = easysimd_test_x86_random_i32x4();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m128 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm_mask_fixupimm_ps, r, easysimd_mm_setzero_ps(), imm8, a, k, b, c);

    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_fixupimm_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const int32_t c[4];
    const int imm8;
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C(124),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   638.62),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -132.35) },
      { EASYSIMD_FLOAT32_C(   380.68),            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -115.28) },
      {  INT32_C(    89279609),  INT32_C(  1064433692),  INT32_C(   831920124),  INT32_C(  1881735017) },
       INT32_C(         173),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    90.00), EASYSIMD_FLOAT32_C(  -132.35) } },
    { UINT8_C(106),
      { EASYSIMD_FLOAT32_C(   -96.63), EASYSIMD_FLOAT32_C(  -711.38), EASYSIMD_FLOAT32_C(   -45.21), EASYSIMD_FLOAT32_C(  -751.31) },
      { EASYSIMD_FLOAT32_C(  -590.46), EASYSIMD_FLOAT32_C(  -177.78), EASYSIMD_FLOAT32_C(  -944.81), EASYSIMD_FLOAT32_C(  -564.35) },
      { -INT32_C(  1439743690),  INT32_C(  1841087159),  INT32_C(   770062561), -INT32_C(  2092307317) },
       INT32_C(          93),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.57), EASYSIMD_FLOAT32_C(     0.00),            EASYSIMD_MATH_NANF } },
    { UINT8_C( 31),
      { EASYSIMD_FLOAT32_C(   787.86), EASYSIMD_FLOAT32_C(   287.26), EASYSIMD_FLOAT32_C(    -7.95), EASYSIMD_FLOAT32_C(  -547.62) },
      { EASYSIMD_FLOAT32_C(   459.87), EASYSIMD_FLOAT32_C(  -641.82), EASYSIMD_FLOAT32_C(   120.92), EASYSIMD_FLOAT32_C(   976.19) },
      { -INT32_C(   652288335),  INT32_C(  1458752334),  INT32_C(   932011254),  INT32_C(  1324234636) },
       INT32_C(         244),
      { EASYSIMD_FLOAT32_C(     1.57),     -EASYSIMD_MATH_INFINITYF,            EASYSIMD_MATH_NANF,     -EASYSIMD_MATH_INFINITYF } },
    { UINT8_C( 66),
      { EASYSIMD_FLOAT32_C(   -62.56), EASYSIMD_FLOAT32_C(   669.91), EASYSIMD_FLOAT32_C(  -683.46), EASYSIMD_FLOAT32_C(   363.85) },
      { EASYSIMD_FLOAT32_C(  -988.00), EASYSIMD_FLOAT32_C(   193.28), EASYSIMD_FLOAT32_C(   703.87), EASYSIMD_FLOAT32_C(  -318.33) },
      { -INT32_C(  1490110627), -INT32_C(  1154512069), -INT32_C(   580104449),  INT32_C(   925582700) },
       INT32_C(          56),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.50), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 15),
      { EASYSIMD_FLOAT32_C(  -168.10), EASYSIMD_FLOAT32_C(   811.95), EASYSIMD_FLOAT32_C(   549.32), EASYSIMD_FLOAT32_C(  -787.42) },
      { EASYSIMD_FLOAT32_C(   257.19), EASYSIMD_FLOAT32_C(  -112.09), EASYSIMD_FLOAT32_C(    97.30), EASYSIMD_FLOAT32_C(  -839.44) },
      {  INT32_C(  1413881957), -INT32_C(  1615906193),  INT32_C(   519893351), -INT32_C(  1436966113) },
       INT32_C(         211),
      {      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(    97.30), EASYSIMD_FLOAT32_C(     1.00) } },
    { UINT8_C(128),
      { EASYSIMD_FLOAT32_C(   176.53), EASYSIMD_FLOAT32_C(  -947.91), EASYSIMD_FLOAT32_C(  -590.75), EASYSIMD_FLOAT32_C(   586.07) },
      { EASYSIMD_FLOAT32_C(  -125.69), EASYSIMD_FLOAT32_C(  -535.56), EASYSIMD_FLOAT32_C(  -978.29), EASYSIMD_FLOAT32_C(  -337.83) },
      { -INT32_C(  1278833017), -INT32_C(   214565179), -INT32_C(  1285995374), -INT32_C(   987583094) },
       INT32_C(          92),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(189),
      { EASYSIMD_FLOAT32_C(   751.71), EASYSIMD_FLOAT32_C(    13.77), EASYSIMD_FLOAT32_C(   114.55), EASYSIMD_FLOAT32_C(   211.58) },
      { EASYSIMD_FLOAT32_C(   371.94), EASYSIMD_FLOAT32_C(  -764.53), EASYSIMD_FLOAT32_C(   187.77), EASYSIMD_FLOAT32_C(  -690.62) },
      { -INT32_C(  1537118902), -INT32_C(  1028115432), -INT32_C(   481740459), -INT32_C(    39191297) },
       INT32_C(          49),
      { EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(     1.57) } },
    { UINT8_C(245),
      { EASYSIMD_FLOAT32_C(   905.38), EASYSIMD_FLOAT32_C(   504.31), EASYSIMD_FLOAT32_C(   673.23), EASYSIMD_FLOAT32_C(   917.38) },
      { EASYSIMD_FLOAT32_C(  -302.41), EASYSIMD_FLOAT32_C(   377.10), EASYSIMD_FLOAT32_C(  -400.95), EASYSIMD_FLOAT32_C(   529.50) },
      {  INT32_C(  1688338498), -INT32_C(   249167931), -INT32_C(  1170480307),  INT32_C(  2027085636) },
       INT32_C(         152),
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00) } },
  };

  easysimd__m128 a, b, r;
  easysimd__m128i c;

  a = easysimd_mm_loadu_ps(test_vec[0].a);
  b = easysimd_mm_loadu_ps(test_vec[0].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[0].c);
  r = easysimd_mm_maskz_fixupimm_ps(test_vec[0].k, a, b, c, INT32_C(         173));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[1].a);
  b = easysimd_mm_loadu_ps(test_vec[1].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[1].c);
  r = easysimd_mm_maskz_fixupimm_ps(test_vec[1].k, a, b, c, INT32_C(          93));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[2].a);
  b = easysimd_mm_loadu_ps(test_vec[2].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[2].c);
  r = easysimd_mm_maskz_fixupimm_ps(test_vec[2].k, a, b, c, INT32_C(         244));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[3].a);
  b = easysimd_mm_loadu_ps(test_vec[3].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[3].c);
  r = easysimd_mm_maskz_fixupimm_ps(test_vec[3].k, a, b, c, INT32_C(          56));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[4].a);
  b = easysimd_mm_loadu_ps(test_vec[4].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[4].c);
  r = easysimd_mm_maskz_fixupimm_ps(test_vec[4].k, a, b, c, INT32_C(         211));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[5].a);
  b = easysimd_mm_loadu_ps(test_vec[5].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[5].c);
  r = easysimd_mm_maskz_fixupimm_ps(test_vec[5].k, a, b, c, INT32_C(          92));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[6].a);
  b = easysimd_mm_loadu_ps(test_vec[6].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[6].c);
  r = easysimd_mm_maskz_fixupimm_ps(test_vec[6].k, a, b, c, INT32_C(          49));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[7].a);
  b = easysimd_mm_loadu_ps(test_vec[7].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[7].c);
  r = easysimd_mm_maskz_fixupimm_ps(test_vec[7].k, a, b, c, INT32_C(         152));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float32 values[8 * 2 * sizeof(easysimd__m128)];
  easysimd_test_x86_random_f32x4_full(8, 2, values, -1000.0f, 1000.0f, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_extract_f32x4(i, 2, 0, values);
    easysimd__m128 b = easysimd_test_x86_random_extract_f32x4(i, 2, 1, values);
    easysimd__m128i c = easysimd_test_x86_random_i32x4();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m128 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm_maskz_fixupimm_ps, r, easysimd_mm_setzero_ps(), imm8, k, a, b, c);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_fixupimm_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[8];
    const easysimd_float32 b[8];
    const int32_t c[8];
    const int imm8;
    const easysimd_float32 r[8];
  } test_vec[] = {
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -717.50),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -862.43),
        EASYSIMD_FLOAT32_C(  -740.31), EASYSIMD_FLOAT32_C(  -981.92), EASYSIMD_FLOAT32_C(   395.85), EASYSIMD_FLOAT32_C(  -866.87) },
      { EASYSIMD_FLOAT32_C(  -920.35),            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -757.57),
        EASYSIMD_FLOAT32_C(   266.74), EASYSIMD_FLOAT32_C(  -725.14), EASYSIMD_FLOAT32_C(   112.06), EASYSIMD_FLOAT32_C(   327.58) },
      {  INT32_C(   778834093),  INT32_C(   354272015), -INT32_C(    78709799), -INT32_C(   165808334),  INT32_C(   871982277),  INT32_C(   444242987),  INT32_C(   453967824), -INT32_C(   692770008) },
       INT32_C(          59),
      { EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(    -1.00),     -EASYSIMD_MATH_INFINITYF,
                   EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(   112.06), EASYSIMD_FLOAT32_C(     1.57) } },
    { { EASYSIMD_FLOAT32_C(  -543.90), EASYSIMD_FLOAT32_C(  -832.07), EASYSIMD_FLOAT32_C(  -172.54), EASYSIMD_FLOAT32_C(  -212.80),
        EASYSIMD_FLOAT32_C(   110.31), EASYSIMD_FLOAT32_C(   254.25), EASYSIMD_FLOAT32_C(   616.87), EASYSIMD_FLOAT32_C(  -886.83) },
      { EASYSIMD_FLOAT32_C(  -780.07), EASYSIMD_FLOAT32_C(   724.14), EASYSIMD_FLOAT32_C(  -604.09), EASYSIMD_FLOAT32_C(  -796.62),
        EASYSIMD_FLOAT32_C(  -291.37), EASYSIMD_FLOAT32_C(   242.20), EASYSIMD_FLOAT32_C(   593.52), EASYSIMD_FLOAT32_C(  -398.18) },
      { -INT32_C(  1100996123),  INT32_C(  1337568796),  INT32_C(  1799739046),  INT32_C(  1721712187), -INT32_C(  1417668134),  INT32_C(  1103531545), -INT32_C(   199787591), -INT32_C(  2109858659) },
       INT32_C(          61),
      { EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(     0.50),     -EASYSIMD_MATH_INFINITYF,
        EASYSIMD_FLOAT32_C(     0.50),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00),            EASYSIMD_MATH_NANF } },
    { { EASYSIMD_FLOAT32_C(   524.71), EASYSIMD_FLOAT32_C(  -255.94), EASYSIMD_FLOAT32_C(  -260.61), EASYSIMD_FLOAT32_C(   784.39),
        EASYSIMD_FLOAT32_C(  -237.86), EASYSIMD_FLOAT32_C(  -864.77), EASYSIMD_FLOAT32_C(   917.52), EASYSIMD_FLOAT32_C(  -158.21) },
      { EASYSIMD_FLOAT32_C(  -793.92), EASYSIMD_FLOAT32_C(  -112.64), EASYSIMD_FLOAT32_C(    84.22), EASYSIMD_FLOAT32_C(   472.81),
        EASYSIMD_FLOAT32_C(   162.22), EASYSIMD_FLOAT32_C(  -803.72), EASYSIMD_FLOAT32_C(  -199.61), EASYSIMD_FLOAT32_C(   618.32) },
      { -INT32_C(   206964403),  INT32_C(   173993679),  INT32_C(   124845357),  INT32_C(   817033239),  INT32_C(   930183550), -INT32_C(  1859417612), -INT32_C(   502043995),  INT32_C(  1430016776) },
       INT32_C(          76),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(  -260.61),            EASYSIMD_MATH_NANF,
                   EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -803.72),            EASYSIMD_MATH_NANF,      EASYSIMD_MATH_INFINITYF } },
    { { EASYSIMD_FLOAT32_C(  -635.79), EASYSIMD_FLOAT32_C(   627.86), EASYSIMD_FLOAT32_C(  -594.48), EASYSIMD_FLOAT32_C(   474.52),
        EASYSIMD_FLOAT32_C(  -117.90), EASYSIMD_FLOAT32_C(  -977.61), EASYSIMD_FLOAT32_C(   587.68), EASYSIMD_FLOAT32_C(   102.04) },
      { EASYSIMD_FLOAT32_C(   746.53), EASYSIMD_FLOAT32_C(   983.59), EASYSIMD_FLOAT32_C(   305.42), EASYSIMD_FLOAT32_C(  -544.84),
        EASYSIMD_FLOAT32_C(   225.79), EASYSIMD_FLOAT32_C(  -101.06), EASYSIMD_FLOAT32_C(    56.98), EASYSIMD_FLOAT32_C(  -249.50) },
      {  INT32_C(    19310548), -INT32_C(  1157064796),  INT32_C(   116112263),  INT32_C(   675110196), -INT32_C(  1950717466), -INT32_C(   613560877),  INT32_C(  1831971361),  INT32_C(  1669953935) },
       INT32_C(          32),
      { EASYSIMD_FLOAT32_C(  -635.79), EASYSIMD_FLOAT32_C(     0.50), EASYSIMD_FLOAT32_C(  -594.48), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.50),      EASYSIMD_MATH_INFINITYF,            EASYSIMD_MATH_NANF } },
    { { EASYSIMD_FLOAT32_C(   643.00), EASYSIMD_FLOAT32_C(   796.36), EASYSIMD_FLOAT32_C(  -465.10), EASYSIMD_FLOAT32_C(  -594.86),
        EASYSIMD_FLOAT32_C(   931.60), EASYSIMD_FLOAT32_C(  -547.58), EASYSIMD_FLOAT32_C(   246.93), EASYSIMD_FLOAT32_C(  -862.33) },
      { EASYSIMD_FLOAT32_C(   339.78), EASYSIMD_FLOAT32_C(  -668.85), EASYSIMD_FLOAT32_C(   610.49), EASYSIMD_FLOAT32_C(  -498.00),
        EASYSIMD_FLOAT32_C(  -472.57), EASYSIMD_FLOAT32_C(  -589.12), EASYSIMD_FLOAT32_C(  -879.69), EASYSIMD_FLOAT32_C(  -108.36) },
      { -INT32_C(   830444217),  INT32_C(  1557425192), -INT32_C(  1350298935),  INT32_C(  1312439931),  INT32_C(   757704460), -INT32_C(   509978031), -INT32_C(   196860716),  INT32_C(   465152468) },
       INT32_C(          22),
      { EASYSIMD_FLOAT32_C(    90.00), EASYSIMD_FLOAT32_C(    90.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00),
        EASYSIMD_FLOAT32_C(     1.57), EASYSIMD_FLOAT32_C(  -589.12),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(     0.50) } },
    { { EASYSIMD_FLOAT32_C(  -961.26), EASYSIMD_FLOAT32_C(  -474.17), EASYSIMD_FLOAT32_C(  -633.84), EASYSIMD_FLOAT32_C(   -79.16),
        EASYSIMD_FLOAT32_C(  -451.78), EASYSIMD_FLOAT32_C(   953.84), EASYSIMD_FLOAT32_C(  -977.12), EASYSIMD_FLOAT32_C(  -705.25) },
      { EASYSIMD_FLOAT32_C(   937.43), EASYSIMD_FLOAT32_C(   328.30), EASYSIMD_FLOAT32_C(  -250.09), EASYSIMD_FLOAT32_C(   163.22),
        EASYSIMD_FLOAT32_C(  -772.76), EASYSIMD_FLOAT32_C(   806.89), EASYSIMD_FLOAT32_C(   913.73), EASYSIMD_FLOAT32_C(   870.24) },
      {  INT32_C(  1872412326),  INT32_C(  1260265168),  INT32_C(  1771657309),  INT32_C(  1368834815), -INT32_C(   248369123),  INT32_C(   686126676),  INT32_C(   893624095),  INT32_C(  2138254809) },
       INT32_C(         235),
      {      EASYSIMD_MATH_INFINITYF,     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(    -1.00),      EASYSIMD_MATH_INFINITYF,
        EASYSIMD_FLOAT32_C(  -772.76),            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    -0.00) } },
    { { EASYSIMD_FLOAT32_C(   603.25), EASYSIMD_FLOAT32_C(  -551.38), EASYSIMD_FLOAT32_C(  -724.63), EASYSIMD_FLOAT32_C(   534.85),
        EASYSIMD_FLOAT32_C(   -98.96), EASYSIMD_FLOAT32_C(   522.30), EASYSIMD_FLOAT32_C(   672.52), EASYSIMD_FLOAT32_C(  -759.19) },
      { EASYSIMD_FLOAT32_C(   853.45), EASYSIMD_FLOAT32_C(   283.01), EASYSIMD_FLOAT32_C(  -257.19), EASYSIMD_FLOAT32_C(  -619.12),
        EASYSIMD_FLOAT32_C(   693.89), EASYSIMD_FLOAT32_C(  -136.88), EASYSIMD_FLOAT32_C(   272.52), EASYSIMD_FLOAT32_C(   732.63) },
      { -INT32_C(  1996092372),  INT32_C(  1676844900),  INT32_C(  2125760609),  INT32_C(   225437368),  INT32_C(  2083870045), -INT32_C(   843941388), -INT32_C(  1857280602),  INT32_C(  1598831155) },
       INT32_C(          70),
      { EASYSIMD_FLOAT32_C(     0.00),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(     1.57),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.57), EASYSIMD_FLOAT32_C(    -1.00),      EASYSIMD_MATH_INFINITYF } },
    { { EASYSIMD_FLOAT32_C(   388.95), EASYSIMD_FLOAT32_C(   638.68), EASYSIMD_FLOAT32_C(  -346.52), EASYSIMD_FLOAT32_C(   937.17),
        EASYSIMD_FLOAT32_C(   592.52), EASYSIMD_FLOAT32_C(  -323.64), EASYSIMD_FLOAT32_C(  -768.07), EASYSIMD_FLOAT32_C(   529.95) },
      { EASYSIMD_FLOAT32_C(  -995.34), EASYSIMD_FLOAT32_C(   -18.16), EASYSIMD_FLOAT32_C(  -306.83), EASYSIMD_FLOAT32_C(  -768.10),
        EASYSIMD_FLOAT32_C(  -211.27), EASYSIMD_FLOAT32_C(  -393.10), EASYSIMD_FLOAT32_C(  -897.87), EASYSIMD_FLOAT32_C(  -608.02) },
      {  INT32_C(  1376639729),  INT32_C(   449954402),  INT32_C(    86458536), -INT32_C(  1987945067),  INT32_C(  2086024406), -INT32_C(  1945263527),  INT32_C(   585849308), -INT32_C(  1664298069) },
       INT32_C(         173),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     1.00),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(    90.00), EASYSIMD_FLOAT32_C(    90.00),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    90.00) } },
  };

  easysimd__m256 a, b, r;
  easysimd__m256i c;

  a = easysimd_mm256_loadu_ps(test_vec[0].a);
  b = easysimd_mm256_loadu_ps(test_vec[0].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[0].c);
  r = easysimd_mm256_fixupimm_ps(a, b, c, INT32_C(          59));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[1].a);
  b = easysimd_mm256_loadu_ps(test_vec[1].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[1].c);
  r = easysimd_mm256_fixupimm_ps(a, b, c, INT32_C(          61));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[2].a);
  b = easysimd_mm256_loadu_ps(test_vec[2].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[2].c);
  r = easysimd_mm256_fixupimm_ps(a, b, c, INT32_C(          76));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[3].a);
  b = easysimd_mm256_loadu_ps(test_vec[3].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[3].c);
  r = easysimd_mm256_fixupimm_ps(a, b, c, INT32_C(          32));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[4].a);
  b = easysimd_mm256_loadu_ps(test_vec[4].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[4].c);
  r = easysimd_mm256_fixupimm_ps(a, b, c, INT32_C(          22));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[5].a);
  b = easysimd_mm256_loadu_ps(test_vec[5].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[5].c);
  r = easysimd_mm256_fixupimm_ps(a, b, c, INT32_C(         235));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[6].a);
  b = easysimd_mm256_loadu_ps(test_vec[6].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[6].c);
  r = easysimd_mm256_fixupimm_ps(a, b, c, INT32_C(          70));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[7].a);
  b = easysimd_mm256_loadu_ps(test_vec[7].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[7].c);
  r = easysimd_mm256_fixupimm_ps(a, b, c, INT32_C(         173));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float32 values[8 * 2 * sizeof(easysimd__m256)];
  easysimd_test_x86_random_f32x8_full(8, 2, values, -1000.0f, 1000.0f, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m256 a = easysimd_test_x86_random_extract_f32x8(i, 2, 0, values);
    easysimd__m256 b = easysimd_test_x86_random_extract_f32x8(i, 2, 1, values);
    easysimd__m256i c = easysimd_test_x86_random_i32x8();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m256 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm256_fixupimm_ps, r, easysimd_mm256_setzero_ps(), imm8, a, b, c);

    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_fixupimm_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[8];
    const easysimd__mmask8 k;
    const easysimd_float32 b[8];
    const int32_t c[8];
    const int imm8;
    const easysimd_float32 r[8];
  } test_vec[] = {
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -701.09),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -192.64),
        EASYSIMD_FLOAT32_C(  -447.73), EASYSIMD_FLOAT32_C(  -514.77), EASYSIMD_FLOAT32_C(  -642.51), EASYSIMD_FLOAT32_C(  -424.83) },
      UINT8_C(180),
      { EASYSIMD_FLOAT32_C(   471.49),            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -677.28),
        EASYSIMD_FLOAT32_C(  -276.38), EASYSIMD_FLOAT32_C(  -462.43), EASYSIMD_FLOAT32_C(  -239.73), EASYSIMD_FLOAT32_C(  -354.30) },
      { -INT32_C(  1317828619), -INT32_C(  1442836490),  INT32_C(   718401157),  INT32_C(  1057338427), -INT32_C(    85827389),  INT32_C(  1211196643), -INT32_C(  1129387323), -INT32_C(  1150287674) },
       INT32_C(         135),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -701.09),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -192.64),
        EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -642.51), EASYSIMD_FLOAT32_C(     0.50) } },
    { { EASYSIMD_FLOAT32_C(  -213.42), EASYSIMD_FLOAT32_C(   235.52), EASYSIMD_FLOAT32_C(  -616.19), EASYSIMD_FLOAT32_C(    64.87),
        EASYSIMD_FLOAT32_C(  -497.47), EASYSIMD_FLOAT32_C(   660.86), EASYSIMD_FLOAT32_C(   -68.87), EASYSIMD_FLOAT32_C(  -443.85) },
      UINT8_C(243),
      { EASYSIMD_FLOAT32_C(   891.56), EASYSIMD_FLOAT32_C(  -909.49), EASYSIMD_FLOAT32_C(  -692.78), EASYSIMD_FLOAT32_C(   691.38),
        EASYSIMD_FLOAT32_C(  -378.80), EASYSIMD_FLOAT32_C(   606.71), EASYSIMD_FLOAT32_C(   445.52), EASYSIMD_FLOAT32_C(  -694.43) },
      {  INT32_C(  1517889644), -INT32_C(  1365924871),  INT32_C(   141677736),  INT32_C(   468413622), -INT32_C(  1998572387), -INT32_C(   498164510), -INT32_C(   244774643), -INT32_C(   488315019) },
       INT32_C(          16),
      {      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(  -616.19), EASYSIMD_FLOAT32_C(    64.87),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00),            EASYSIMD_MATH_NANF } },
    { { EASYSIMD_FLOAT32_C(   905.62), EASYSIMD_FLOAT32_C(   -67.52), EASYSIMD_FLOAT32_C(   112.93), EASYSIMD_FLOAT32_C(  -542.11),
        EASYSIMD_FLOAT32_C(   417.71), EASYSIMD_FLOAT32_C(   470.42), EASYSIMD_FLOAT32_C(    33.06), EASYSIMD_FLOAT32_C(  -110.80) },
      UINT8_C(  0),
      { EASYSIMD_FLOAT32_C(  -604.83), EASYSIMD_FLOAT32_C(  -390.12), EASYSIMD_FLOAT32_C(   211.92), EASYSIMD_FLOAT32_C(   118.79),
        EASYSIMD_FLOAT32_C(   147.45), EASYSIMD_FLOAT32_C(   972.19), EASYSIMD_FLOAT32_C(   764.49), EASYSIMD_FLOAT32_C(   934.03) },
      { -INT32_C(  1498892334), -INT32_C(  1789022167), -INT32_C(   801998692), -INT32_C(   172836264),  INT32_C(   285381640), -INT32_C(   444075011),  INT32_C(   905275863),  INT32_C(  2000027301) },
       INT32_C(         182),
      { EASYSIMD_FLOAT32_C(   905.62), EASYSIMD_FLOAT32_C(   -67.52), EASYSIMD_FLOAT32_C(   112.93), EASYSIMD_FLOAT32_C(  -542.11),
        EASYSIMD_FLOAT32_C(   417.71), EASYSIMD_FLOAT32_C(   470.42), EASYSIMD_FLOAT32_C(    33.06), EASYSIMD_FLOAT32_C(  -110.80) } },
    { { EASYSIMD_FLOAT32_C(   207.71), EASYSIMD_FLOAT32_C(  -851.70), EASYSIMD_FLOAT32_C(    -1.10), EASYSIMD_FLOAT32_C(   710.23),
        EASYSIMD_FLOAT32_C(   809.17), EASYSIMD_FLOAT32_C(   930.03), EASYSIMD_FLOAT32_C(  -733.61), EASYSIMD_FLOAT32_C(   700.73) },
      UINT8_C(142),
      { EASYSIMD_FLOAT32_C(  -979.45), EASYSIMD_FLOAT32_C(  -426.40), EASYSIMD_FLOAT32_C(   392.11), EASYSIMD_FLOAT32_C(  -358.25),
        EASYSIMD_FLOAT32_C(  -819.69), EASYSIMD_FLOAT32_C(  -162.38), EASYSIMD_FLOAT32_C(   -52.68), EASYSIMD_FLOAT32_C(  -914.07) },
      { -INT32_C(   215321477),  INT32_C(  1632369318),  INT32_C(  1080639660),  INT32_C(   893287234),  INT32_C(  1796023042),  INT32_C(   370164248),  INT32_C(  1439467639),  INT32_C(   568634278) },
       INT32_C(          31),
      { EASYSIMD_FLOAT32_C(   207.71), EASYSIMD_FLOAT32_C(  -426.40),     -EASYSIMD_MATH_INFINITYF,      EASYSIMD_MATH_INFINITYF,
        EASYSIMD_FLOAT32_C(   809.17), EASYSIMD_FLOAT32_C(   930.03), EASYSIMD_FLOAT32_C(  -733.61), EASYSIMD_FLOAT32_C(  -914.07) } },
    { { EASYSIMD_FLOAT32_C(   770.10), EASYSIMD_FLOAT32_C(  -939.75), EASYSIMD_FLOAT32_C(  -456.19), EASYSIMD_FLOAT32_C(   187.81),
        EASYSIMD_FLOAT32_C(   530.68), EASYSIMD_FLOAT32_C(   576.87), EASYSIMD_FLOAT32_C(  -922.99), EASYSIMD_FLOAT32_C(   925.85) },
      UINT8_C(  9),
      { EASYSIMD_FLOAT32_C(  -813.24), EASYSIMD_FLOAT32_C(   288.94), EASYSIMD_FLOAT32_C(    44.64), EASYSIMD_FLOAT32_C(   334.21),
        EASYSIMD_FLOAT32_C(   261.13), EASYSIMD_FLOAT32_C(  -190.87), EASYSIMD_FLOAT32_C(   268.25), EASYSIMD_FLOAT32_C(  -531.16) },
      { -INT32_C(  1598740641),  INT32_C(   199423632),  INT32_C(  1460475956),  INT32_C(  1735358501),  INT32_C(   299795849), -INT32_C(    38325166),  INT32_C(   639490072),  INT32_C(  1261429740) },
       INT32_C(          10),
      { EASYSIMD_FLOAT32_C(   770.10), EASYSIMD_FLOAT32_C(  -939.75), EASYSIMD_FLOAT32_C(  -456.19),      EASYSIMD_MATH_INFINITYF,
        EASYSIMD_FLOAT32_C(   530.68), EASYSIMD_FLOAT32_C(   576.87), EASYSIMD_FLOAT32_C(  -922.99), EASYSIMD_FLOAT32_C(   925.85) } },
    { { EASYSIMD_FLOAT32_C(   -42.56), EASYSIMD_FLOAT32_C(  -732.85), EASYSIMD_FLOAT32_C(  -820.93), EASYSIMD_FLOAT32_C(  -233.39),
        EASYSIMD_FLOAT32_C(  -802.82), EASYSIMD_FLOAT32_C(  -554.54), EASYSIMD_FLOAT32_C(  -532.66), EASYSIMD_FLOAT32_C(  -782.27) },
      UINT8_C(219),
      { EASYSIMD_FLOAT32_C(    19.06), EASYSIMD_FLOAT32_C(   859.44), EASYSIMD_FLOAT32_C(  -140.53), EASYSIMD_FLOAT32_C(   199.37),
        EASYSIMD_FLOAT32_C(  -302.94), EASYSIMD_FLOAT32_C(   806.79), EASYSIMD_FLOAT32_C(   285.30), EASYSIMD_FLOAT32_C(  -532.83) },
      { -INT32_C(   435181874),  INT32_C(   738944691),  INT32_C(  1521840853), -INT32_C(  2069051824),  INT32_C(  1436330621), -INT32_C(  1438530617), -INT32_C(   676033294),  INT32_C(  1186090616) },
       INT32_C(         244),
      { EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -820.93), EASYSIMD_FLOAT32_C(     0.00),
             EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -554.54), EASYSIMD_FLOAT32_C(     1.57),     -EASYSIMD_MATH_INFINITYF } },
    { { EASYSIMD_FLOAT32_C(   867.05), EASYSIMD_FLOAT32_C(   829.11), EASYSIMD_FLOAT32_C(   654.98), EASYSIMD_FLOAT32_C(   397.72),
        EASYSIMD_FLOAT32_C(   405.99), EASYSIMD_FLOAT32_C(   731.99), EASYSIMD_FLOAT32_C(   323.57), EASYSIMD_FLOAT32_C(   592.74) },
      UINT8_C( 39),
      { EASYSIMD_FLOAT32_C(    20.93), EASYSIMD_FLOAT32_C(  -631.79), EASYSIMD_FLOAT32_C(   -73.04), EASYSIMD_FLOAT32_C(  -717.94),
        EASYSIMD_FLOAT32_C(   177.34), EASYSIMD_FLOAT32_C(  -804.80), EASYSIMD_FLOAT32_C(  -249.11), EASYSIMD_FLOAT32_C(  -865.22) },
      { -INT32_C(  1443048393),  INT32_C(  1341740937), -INT32_C(  1379107325), -INT32_C(   579591910), -INT32_C(   288350622),  INT32_C(   560375762),  INT32_C(   454405210),  INT32_C(   256097752) },
       INT32_C(         145),
      { EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(     1.57), EASYSIMD_FLOAT32_C(   397.72),
        EASYSIMD_FLOAT32_C(   405.99), EASYSIMD_FLOAT32_C(  -804.80), EASYSIMD_FLOAT32_C(   323.57), EASYSIMD_FLOAT32_C(   592.74) } },
    { { EASYSIMD_FLOAT32_C(  -537.65), EASYSIMD_FLOAT32_C(   -70.04), EASYSIMD_FLOAT32_C(   -98.61), EASYSIMD_FLOAT32_C(  -340.47),
        EASYSIMD_FLOAT32_C(   375.42), EASYSIMD_FLOAT32_C(   368.72), EASYSIMD_FLOAT32_C(  -122.75), EASYSIMD_FLOAT32_C(  -605.52) },
      UINT8_C(150),
      { EASYSIMD_FLOAT32_C(   228.17), EASYSIMD_FLOAT32_C(   736.72), EASYSIMD_FLOAT32_C(   593.85), EASYSIMD_FLOAT32_C(   925.23),
        EASYSIMD_FLOAT32_C(   543.52), EASYSIMD_FLOAT32_C(  -120.85), EASYSIMD_FLOAT32_C(  -607.60), EASYSIMD_FLOAT32_C(   410.57) },
      {  INT32_C(   815425970),  INT32_C(  1447708469), -INT32_C(   625465156), -INT32_C(  1616009224), -INT32_C(  1158033907), -INT32_C(  1584261661), -INT32_C(  1758289320),  INT32_C(   204361050) },
       INT32_C(         182),
      { EASYSIMD_FLOAT32_C(  -537.65),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(     1.57), EASYSIMD_FLOAT32_C(  -340.47),
        EASYSIMD_FLOAT32_C(     0.50), EASYSIMD_FLOAT32_C(   368.72), EASYSIMD_FLOAT32_C(  -122.75), EASYSIMD_FLOAT32_C(  -605.52) } },
  };

  easysimd__m256 a, b, r;
  easysimd__m256i c;

  a = easysimd_mm256_loadu_ps(test_vec[0].a);
  b = easysimd_mm256_loadu_ps(test_vec[0].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[0].c);
  r = easysimd_mm256_mask_fixupimm_ps(a, test_vec[0].k, b, c, INT32_C(         135));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[1].a);
  b = easysimd_mm256_loadu_ps(test_vec[1].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[1].c);
  r = easysimd_mm256_mask_fixupimm_ps(a, test_vec[1].k, b, c, INT32_C(          16));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[2].a);
  b = easysimd_mm256_loadu_ps(test_vec[2].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[2].c);
  r = easysimd_mm256_mask_fixupimm_ps(a, test_vec[2].k, b, c, INT32_C(         182));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[3].a);
  b = easysimd_mm256_loadu_ps(test_vec[3].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[3].c);
  r = easysimd_mm256_mask_fixupimm_ps(a, test_vec[3].k, b, c, INT32_C(          31));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[4].a);
  b = easysimd_mm256_loadu_ps(test_vec[4].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[4].c);
  r = easysimd_mm256_mask_fixupimm_ps(a, test_vec[4].k, b, c, INT32_C(          10));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[5].a);
  b = easysimd_mm256_loadu_ps(test_vec[5].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[5].c);
  r = easysimd_mm256_mask_fixupimm_ps(a, test_vec[5].k, b, c, INT32_C(         244));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[6].a);
  b = easysimd_mm256_loadu_ps(test_vec[6].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[6].c);
  r = easysimd_mm256_mask_fixupimm_ps(a, test_vec[6].k, b, c, INT32_C(         145));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[7].a);
  b = easysimd_mm256_loadu_ps(test_vec[7].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[7].c);
  r = easysimd_mm256_mask_fixupimm_ps(a, test_vec[7].k, b, c, INT32_C(         182));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float32 values[8 * 2 * sizeof(easysimd__m256)];
  easysimd_test_x86_random_f32x8_full(8, 2, values, -1000.0f, 1000.0f, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m256 a = easysimd_test_x86_random_extract_f32x8(i, 2, 0, values);
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 b = easysimd_test_x86_random_extract_f32x8(i, 2, 1, values);
    easysimd__m256i c = easysimd_test_x86_random_i32x8();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m256 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm256_mask_fixupimm_ps, r, easysimd_mm256_setzero_ps(), imm8, a, k, b, c);

    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_fixupimm_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float32 a[8];
    const easysimd_float32 b[8];
    const int32_t c[8];
    const int imm8;
    const easysimd_float32 r[8];
  } test_vec[] = {
    { UINT8_C( 69),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   940.69),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -989.43),
        EASYSIMD_FLOAT32_C(  -775.07), EASYSIMD_FLOAT32_C(  -593.92), EASYSIMD_FLOAT32_C(  -620.41), EASYSIMD_FLOAT32_C(    43.35) },
      { EASYSIMD_FLOAT32_C(  -596.35),            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   277.21),
        EASYSIMD_FLOAT32_C(  -589.86), EASYSIMD_FLOAT32_C(    41.85), EASYSIMD_FLOAT32_C(  -572.85), EASYSIMD_FLOAT32_C(   447.23) },
      { -INT32_C(   266951678),  INT32_C(  1046758520), -INT32_C(  1658184077), -INT32_C(   227928870), -INT32_C(   776419921), -INT32_C(   668170996),  INT32_C(  1544174320),  INT32_C(  2141363325) },
       INT32_C(          64),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     0.00),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    90.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(252),
      { EASYSIMD_FLOAT32_C(    50.98), EASYSIMD_FLOAT32_C(   -52.38), EASYSIMD_FLOAT32_C(  -835.14), EASYSIMD_FLOAT32_C(  -418.31),
        EASYSIMD_FLOAT32_C(   915.06), EASYSIMD_FLOAT32_C(   249.74), EASYSIMD_FLOAT32_C(   801.60), EASYSIMD_FLOAT32_C(  -744.22) },
      { EASYSIMD_FLOAT32_C(   225.98), EASYSIMD_FLOAT32_C(  -157.56), EASYSIMD_FLOAT32_C(   113.96), EASYSIMD_FLOAT32_C(   471.85),
        EASYSIMD_FLOAT32_C(   608.79), EASYSIMD_FLOAT32_C(   917.82), EASYSIMD_FLOAT32_C(  -800.62), EASYSIMD_FLOAT32_C(  -150.47) },
      { -INT32_C(   244320301),  INT32_C(   902499617),  INT32_C(  1038466423), -INT32_C(    79055242),  INT32_C(   501948899), -INT32_C(   929413076), -INT32_C(  1576527126),  INT32_C(  1570750601) },
       INT32_C(         183),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00),
        EASYSIMD_FLOAT32_C(   608.79), EASYSIMD_FLOAT32_C(    90.00),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     1.57) } },
    { UINT8_C( 27),
      { EASYSIMD_FLOAT32_C(   858.50), EASYSIMD_FLOAT32_C(  -868.27), EASYSIMD_FLOAT32_C(  -139.90), EASYSIMD_FLOAT32_C(  -916.57),
        EASYSIMD_FLOAT32_C(  -462.20), EASYSIMD_FLOAT32_C(   239.69), EASYSIMD_FLOAT32_C(   126.79), EASYSIMD_FLOAT32_C(   -58.54) },
      { EASYSIMD_FLOAT32_C(   974.53), EASYSIMD_FLOAT32_C(    64.32), EASYSIMD_FLOAT32_C(  -781.34), EASYSIMD_FLOAT32_C(  -615.33),
        EASYSIMD_FLOAT32_C(  -893.82), EASYSIMD_FLOAT32_C(  -354.19), EASYSIMD_FLOAT32_C(   831.90), EASYSIMD_FLOAT32_C(   157.16) },
      { -INT32_C(   678228454), -INT32_C(  1454518029),  INT32_C(   999049241),  INT32_C(  2087168308),  INT32_C(  1583755076), -INT32_C(   135854025), -INT32_C(  1230027609), -INT32_C(  1378777197) },
       INT32_C(         149),
      { EASYSIMD_FLOAT32_C(     1.57), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    90.00),
        EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C( 52),
      { EASYSIMD_FLOAT32_C(   593.43), EASYSIMD_FLOAT32_C(   996.76), EASYSIMD_FLOAT32_C(   738.85), EASYSIMD_FLOAT32_C(   508.50),
        EASYSIMD_FLOAT32_C(   246.50), EASYSIMD_FLOAT32_C(   540.45), EASYSIMD_FLOAT32_C(   764.28), EASYSIMD_FLOAT32_C(  -527.52) },
      { EASYSIMD_FLOAT32_C(  -617.11), EASYSIMD_FLOAT32_C(  -121.76), EASYSIMD_FLOAT32_C(   944.33), EASYSIMD_FLOAT32_C(   991.68),
        EASYSIMD_FLOAT32_C(  -203.94), EASYSIMD_FLOAT32_C(  -856.29), EASYSIMD_FLOAT32_C(  -158.79), EASYSIMD_FLOAT32_C(  -345.44) },
      {  INT32_C(   441332434),  INT32_C(  1749977534),  INT32_C(   531417840),  INT32_C(   961940016),  INT32_C(   920669681), -INT32_C(  2067163396), -INT32_C(   870746520), -INT32_C(   872307974) },
       INT32_C(         212),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   944.33), EASYSIMD_FLOAT32_C(     0.00),
            -EASYSIMD_MATH_INFINITYF,     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(216),
      { EASYSIMD_FLOAT32_C(  -724.56), EASYSIMD_FLOAT32_C(   701.31), EASYSIMD_FLOAT32_C(  -262.00), EASYSIMD_FLOAT32_C(  -186.76),
        EASYSIMD_FLOAT32_C(   -59.01), EASYSIMD_FLOAT32_C(   864.78), EASYSIMD_FLOAT32_C(   754.70), EASYSIMD_FLOAT32_C(   -84.47) },
      { EASYSIMD_FLOAT32_C(   -70.90), EASYSIMD_FLOAT32_C(   973.36), EASYSIMD_FLOAT32_C(   300.20), EASYSIMD_FLOAT32_C(    35.28),
        EASYSIMD_FLOAT32_C(  -380.82), EASYSIMD_FLOAT32_C(   132.10), EASYSIMD_FLOAT32_C(  -807.56), EASYSIMD_FLOAT32_C(  -787.39) },
      {  INT32_C(    13171253), -INT32_C(  1338972250), -INT32_C(  1969067715), -INT32_C(   527968182),  INT32_C(   390597537), -INT32_C(   971959004),  INT32_C(  1704648214), -INT32_C(    96653883) },
       INT32_C(          39),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(     1.00) } },
    { UINT8_C(236),
      { EASYSIMD_FLOAT32_C(   128.86), EASYSIMD_FLOAT32_C(   931.29), EASYSIMD_FLOAT32_C(   721.11), EASYSIMD_FLOAT32_C(  -624.64),
        EASYSIMD_FLOAT32_C(   471.74), EASYSIMD_FLOAT32_C(   485.39), EASYSIMD_FLOAT32_C(  -152.15), EASYSIMD_FLOAT32_C(   854.63) },
      { EASYSIMD_FLOAT32_C(  -636.37), EASYSIMD_FLOAT32_C(  -207.82), EASYSIMD_FLOAT32_C(   846.32), EASYSIMD_FLOAT32_C(   159.69),
        EASYSIMD_FLOAT32_C(   -64.11), EASYSIMD_FLOAT32_C(  -312.47), EASYSIMD_FLOAT32_C(   814.25), EASYSIMD_FLOAT32_C(   211.33) },
      { -INT32_C(  1809220053), -INT32_C(   119622880),  INT32_C(  1167703866),  INT32_C(   476753927), -INT32_C(  1607323454), -INT32_C(   127559733), -INT32_C(   652255276), -INT32_C(  2050626214) },
       INT32_C(         108),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   159.69),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.57), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(162),
      { EASYSIMD_FLOAT32_C(  -611.17), EASYSIMD_FLOAT32_C(  -447.75), EASYSIMD_FLOAT32_C(  -975.43), EASYSIMD_FLOAT32_C(   329.83),
        EASYSIMD_FLOAT32_C(  -582.97), EASYSIMD_FLOAT32_C(   779.27), EASYSIMD_FLOAT32_C(  -754.65), EASYSIMD_FLOAT32_C(   346.13) },
      { EASYSIMD_FLOAT32_C(   752.63), EASYSIMD_FLOAT32_C(   545.55), EASYSIMD_FLOAT32_C(  -618.59), EASYSIMD_FLOAT32_C(  -628.19),
        EASYSIMD_FLOAT32_C(  -322.35), EASYSIMD_FLOAT32_C(  -426.15), EASYSIMD_FLOAT32_C(  -415.58), EASYSIMD_FLOAT32_C(   806.52) },
      { -INT32_C(  1210284552), -INT32_C(   809623010), -INT32_C(  1131291764), -INT32_C(  1517866739), -INT32_C(   176586858), -INT32_C(  1940958305),  INT32_C(   133747736), -INT32_C(   425097746) },
       INT32_C(          10),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    90.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    90.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00) } },
    { UINT8_C(167),
      { EASYSIMD_FLOAT32_C(  -494.85), EASYSIMD_FLOAT32_C(  -694.47), EASYSIMD_FLOAT32_C(  -818.12), EASYSIMD_FLOAT32_C(   976.89),
        EASYSIMD_FLOAT32_C(   790.92), EASYSIMD_FLOAT32_C(    29.73), EASYSIMD_FLOAT32_C(   831.52), EASYSIMD_FLOAT32_C(  -845.45) },
      { EASYSIMD_FLOAT32_C(   821.90), EASYSIMD_FLOAT32_C(   677.84), EASYSIMD_FLOAT32_C(   314.24), EASYSIMD_FLOAT32_C(  -242.21),
        EASYSIMD_FLOAT32_C(  -634.64), EASYSIMD_FLOAT32_C(   128.49), EASYSIMD_FLOAT32_C(   969.12), EASYSIMD_FLOAT32_C(  -245.80) },
      {  INT32_C(   909375323), -INT32_C(  1186664312), -INT32_C(   145757833), -INT32_C(  1248443038),  INT32_C(  1775116948), -INT32_C(  1604856549),  INT32_C(    61488510),  INT32_C(   900453082) },
       INT32_C(         202),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     0.50), EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00),      EASYSIMD_MATH_INFINITYF } },
  };

  easysimd__m256 a, b, r;
  easysimd__m256i c;

  a = easysimd_mm256_loadu_ps(test_vec[0].a);
  b = easysimd_mm256_loadu_ps(test_vec[0].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[0].c);
  r = easysimd_mm256_maskz_fixupimm_ps(test_vec[0].k, a, b, c, INT32_C(          64));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[1].a);
  b = easysimd_mm256_loadu_ps(test_vec[1].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[1].c);
  r = easysimd_mm256_maskz_fixupimm_ps(test_vec[1].k, a, b, c, INT32_C(         183));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[2].a);
  b = easysimd_mm256_loadu_ps(test_vec[2].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[2].c);
  r = easysimd_mm256_maskz_fixupimm_ps(test_vec[2].k, a, b, c, INT32_C(         149));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[3].a);
  b = easysimd_mm256_loadu_ps(test_vec[3].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[3].c);
  r = easysimd_mm256_maskz_fixupimm_ps(test_vec[3].k, a, b, c, INT32_C(         212));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[4].a);
  b = easysimd_mm256_loadu_ps(test_vec[4].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[4].c);
  r = easysimd_mm256_maskz_fixupimm_ps(test_vec[4].k, a, b, c, INT32_C(          39));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[5].a);
  b = easysimd_mm256_loadu_ps(test_vec[5].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[5].c);
  r = easysimd_mm256_maskz_fixupimm_ps(test_vec[5].k, a, b, c, INT32_C(         108));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[6].a);
  b = easysimd_mm256_loadu_ps(test_vec[6].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[6].c);
  r = easysimd_mm256_maskz_fixupimm_ps(test_vec[6].k, a, b, c, INT32_C(          10));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[7].a);
  b = easysimd_mm256_loadu_ps(test_vec[7].b);
  c = easysimd_x_mm256_loadu_epi32(test_vec[7].c);
  r = easysimd_mm256_maskz_fixupimm_ps(test_vec[7].k, a, b, c, INT32_C(         202));
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float32 values[8 * 2 * sizeof(easysimd__m256)];
  easysimd_test_x86_random_f32x8_full(8, 2, values, -1000.0f, 1000.0f, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_extract_f32x8(i, 2, 0, values);
    easysimd__m256 b = easysimd_test_x86_random_extract_f32x8(i, 2, 1, values);
    easysimd__m256i c = easysimd_test_x86_random_i32x8();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m256 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm256_maskz_fixupimm_ps, r, easysimd_mm256_setzero_ps(), imm8, k, a, b, c);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_fixupimm_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const int32_t c[16];
    const int imm8;
    const easysimd_float32 r[16];
  } test_vec[] = {
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -653.08),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -140.64),
        EASYSIMD_FLOAT32_C(  -825.82), EASYSIMD_FLOAT32_C(  -279.01), EASYSIMD_FLOAT32_C(   902.12), EASYSIMD_FLOAT32_C(  -376.10),
        EASYSIMD_FLOAT32_C(   812.35), EASYSIMD_FLOAT32_C(  -490.82), EASYSIMD_FLOAT32_C(  -666.64), EASYSIMD_FLOAT32_C(  -639.79),
        EASYSIMD_FLOAT32_C(   715.91), EASYSIMD_FLOAT32_C(  -962.06), EASYSIMD_FLOAT32_C(   524.79), EASYSIMD_FLOAT32_C(   -57.63) },
      { EASYSIMD_FLOAT32_C(  -611.63),            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -962.93),
        EASYSIMD_FLOAT32_C(  -810.24), EASYSIMD_FLOAT32_C(  -361.16), EASYSIMD_FLOAT32_C(  -655.64), EASYSIMD_FLOAT32_C(  -360.58),
        EASYSIMD_FLOAT32_C(   652.40), EASYSIMD_FLOAT32_C(  -561.63), EASYSIMD_FLOAT32_C(  -896.90), EASYSIMD_FLOAT32_C(   595.45),
        EASYSIMD_FLOAT32_C(    59.08), EASYSIMD_FLOAT32_C(   135.17), EASYSIMD_FLOAT32_C(  -174.98), EASYSIMD_FLOAT32_C(   291.83) },
      {  INT32_C(    65344024), -INT32_C(  1043000603),  INT32_C(   310236860),  INT32_C(  1073926926),  INT32_C(  1299399004),  INT32_C(   912095120), -INT32_C(  1084049800),  INT32_C(   855801371),
         INT32_C(  1966532752), -INT32_C(  1237971974), -INT32_C(   271993631), -INT32_C(   483406969), -INT32_C(  1674534388), -INT32_C(  1831629286),  INT32_C(  1599157572),  INT32_C(  1150440627) },
       INT32_C(          60),
      {            EASYSIMD_MATH_NANF,      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(    90.00), EASYSIMD_FLOAT32_C(  -140.64),
        EASYSIMD_FLOAT32_C(     1.57),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00),            EASYSIMD_MATH_NANF,
        EASYSIMD_FLOAT32_C(    -0.00),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00),     -EASYSIMD_MATH_INFINITYF } },
    { { EASYSIMD_FLOAT32_C(   482.09), EASYSIMD_FLOAT32_C(   735.25), EASYSIMD_FLOAT32_C(  -848.81), EASYSIMD_FLOAT32_C(   656.27),
        EASYSIMD_FLOAT32_C(  -543.76), EASYSIMD_FLOAT32_C(  -946.69), EASYSIMD_FLOAT32_C(  -719.83), EASYSIMD_FLOAT32_C(  -731.41),
        EASYSIMD_FLOAT32_C(  -437.51), EASYSIMD_FLOAT32_C(  -386.47), EASYSIMD_FLOAT32_C(  -371.20), EASYSIMD_FLOAT32_C(  -721.60),
        EASYSIMD_FLOAT32_C(  -348.52), EASYSIMD_FLOAT32_C(  -846.41), EASYSIMD_FLOAT32_C(   220.77), EASYSIMD_FLOAT32_C(    39.85) },
      { EASYSIMD_FLOAT32_C(   463.28), EASYSIMD_FLOAT32_C(   704.75), EASYSIMD_FLOAT32_C(    76.92), EASYSIMD_FLOAT32_C(   653.03),
        EASYSIMD_FLOAT32_C(  -656.41), EASYSIMD_FLOAT32_C(   421.28), EASYSIMD_FLOAT32_C(  -707.55), EASYSIMD_FLOAT32_C(   995.99),
        EASYSIMD_FLOAT32_C(   859.65), EASYSIMD_FLOAT32_C(  -604.45), EASYSIMD_FLOAT32_C(   591.44), EASYSIMD_FLOAT32_C(   -81.27),
        EASYSIMD_FLOAT32_C(   530.72), EASYSIMD_FLOAT32_C(  -583.54), EASYSIMD_FLOAT32_C(  -789.43), EASYSIMD_FLOAT32_C(    12.82) },
      { -INT32_C(  1259540269),  INT32_C(   732149156), -INT32_C(  1928408191), -INT32_C(  1893056907),  INT32_C(   287505868), -INT32_C(   445615310),  INT32_C(    53019591), -INT32_C(  1623596085),
         INT32_C(  2001938131),  INT32_C(  1554184155),  INT32_C(  1072345290), -INT32_C(  1144056594),  INT32_C(  1137504529),  INT32_C(   740834404),  INT32_C(   187650623),  INT32_C(   145385781) },
       INT32_C(         143),
      { EASYSIMD_FLOAT32_C(     0.50),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  -656.41), EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(    90.00),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     0.50),
            -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(    90.00), EASYSIMD_FLOAT32_C(     0.50), EASYSIMD_FLOAT32_C(    39.85) } },
    { { EASYSIMD_FLOAT32_C(  -848.28), EASYSIMD_FLOAT32_C(  -638.24), EASYSIMD_FLOAT32_C(  -330.91), EASYSIMD_FLOAT32_C(  -392.04),
        EASYSIMD_FLOAT32_C(  -584.93), EASYSIMD_FLOAT32_C(   -50.74), EASYSIMD_FLOAT32_C(  -123.45), EASYSIMD_FLOAT32_C(   -22.44),
        EASYSIMD_FLOAT32_C(   562.79), EASYSIMD_FLOAT32_C(   505.35), EASYSIMD_FLOAT32_C(   255.96), EASYSIMD_FLOAT32_C(  -785.73),
        EASYSIMD_FLOAT32_C(   658.94), EASYSIMD_FLOAT32_C(  -523.27), EASYSIMD_FLOAT32_C(   254.12), EASYSIMD_FLOAT32_C(   122.21) },
      { EASYSIMD_FLOAT32_C(  -818.52), EASYSIMD_FLOAT32_C(  -668.97), EASYSIMD_FLOAT32_C(  -224.75), EASYSIMD_FLOAT32_C(  -474.93),
        EASYSIMD_FLOAT32_C(   752.31), EASYSIMD_FLOAT32_C(    67.70), EASYSIMD_FLOAT32_C(  -478.94), EASYSIMD_FLOAT32_C(   611.97),
        EASYSIMD_FLOAT32_C(   463.25), EASYSIMD_FLOAT32_C(  -887.50), EASYSIMD_FLOAT32_C(  -469.30), EASYSIMD_FLOAT32_C(    -6.03),
        EASYSIMD_FLOAT32_C(  -471.04), EASYSIMD_FLOAT32_C(  -258.73), EASYSIMD_FLOAT32_C(  -993.21), EASYSIMD_FLOAT32_C(  -319.32) },
      { -INT32_C(  1094245900), -INT32_C(  1073827375), -INT32_C(   696595003),  INT32_C(   572081854), -INT32_C(  1035058813), -INT32_C(   926056813),  INT32_C(  1993373671),  INT32_C(  1776308085),
         INT32_C(  1093183344),  INT32_C(   486614616), -INT32_C(  1309443085),  INT32_C(  1171459266), -INT32_C(   519626162), -INT32_C(  2018847328), -INT32_C(  1040352692),  INT32_C(   975953354) },
       INT32_C(         133),
      { EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00),     -EASYSIMD_MATH_INFINITYF,            EASYSIMD_MATH_NANF,
        EASYSIMD_FLOAT32_C(    90.00), EASYSIMD_FLOAT32_C(    90.00),     -EASYSIMD_MATH_INFINITYF,      EASYSIMD_MATH_INFINITYF,
            -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(     1.57), EASYSIMD_FLOAT32_C(  -469.30),      EASYSIMD_MATH_INFINITYF,
        EASYSIMD_FLOAT32_C(  -471.04), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(  -993.21), EASYSIMD_FLOAT32_C(     1.00) } },
    { { EASYSIMD_FLOAT32_C(   103.02), EASYSIMD_FLOAT32_C(  -324.12), EASYSIMD_FLOAT32_C(   288.63), EASYSIMD_FLOAT32_C(   518.09),
        EASYSIMD_FLOAT32_C(   625.14), EASYSIMD_FLOAT32_C(  -834.82), EASYSIMD_FLOAT32_C(  -504.35), EASYSIMD_FLOAT32_C(   187.94),
        EASYSIMD_FLOAT32_C(   670.53), EASYSIMD_FLOAT32_C(   751.61), EASYSIMD_FLOAT32_C(   402.21), EASYSIMD_FLOAT32_C(   329.47),
        EASYSIMD_FLOAT32_C(  -771.66), EASYSIMD_FLOAT32_C(  -343.67), EASYSIMD_FLOAT32_C(  -548.31), EASYSIMD_FLOAT32_C(  -590.18) },
      { EASYSIMD_FLOAT32_C(   -12.64), EASYSIMD_FLOAT32_C(   226.93), EASYSIMD_FLOAT32_C(   -65.11), EASYSIMD_FLOAT32_C(  -260.33),
        EASYSIMD_FLOAT32_C(  -705.37), EASYSIMD_FLOAT32_C(   455.95), EASYSIMD_FLOAT32_C(  -648.36), EASYSIMD_FLOAT32_C(   757.88),
        EASYSIMD_FLOAT32_C(   568.45), EASYSIMD_FLOAT32_C(  -117.66), EASYSIMD_FLOAT32_C(  -248.14), EASYSIMD_FLOAT32_C(  -902.59),
        EASYSIMD_FLOAT32_C(   623.60), EASYSIMD_FLOAT32_C(  -241.35), EASYSIMD_FLOAT32_C(  -221.92), EASYSIMD_FLOAT32_C(  -273.37) },
      {  INT32_C(  1828420985), -INT32_C(  1172443400),  INT32_C(  1241510139), -INT32_C(  1305868526),  INT32_C(   674878684),  INT32_C(   434779727), -INT32_C(  1722608364), -INT32_C(   512307352),
         INT32_C(  1162703180),  INT32_C(  1543465568),  INT32_C(  1839529818), -INT32_C(   501232122), -INT32_C(   234202717), -INT32_C(  1559497585),  INT32_C(  1899781641),  INT32_C(  2052240174) },
       INT32_C(          36),
      { EASYSIMD_FLOAT32_C(    90.00), EASYSIMD_FLOAT32_C(     0.50), EASYSIMD_FLOAT32_C(    -1.00),            EASYSIMD_MATH_NANF,
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   455.95), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00),
            -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(     0.50), EASYSIMD_FLOAT32_C(     1.57),            EASYSIMD_MATH_NANF,
        EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -221.92), EASYSIMD_FLOAT32_C(     1.00) } },
    { { EASYSIMD_FLOAT32_C(   434.53), EASYSIMD_FLOAT32_C(  -933.28), EASYSIMD_FLOAT32_C(  -755.28), EASYSIMD_FLOAT32_C(    59.68),
        EASYSIMD_FLOAT32_C(  -768.10), EASYSIMD_FLOAT32_C(  -259.62), EASYSIMD_FLOAT32_C(  -752.38), EASYSIMD_FLOAT32_C(   902.43),
        EASYSIMD_FLOAT32_C(  -508.01), EASYSIMD_FLOAT32_C(   649.83), EASYSIMD_FLOAT32_C(   231.90), EASYSIMD_FLOAT32_C(  -279.67),
        EASYSIMD_FLOAT32_C(  -693.85), EASYSIMD_FLOAT32_C(   683.59), EASYSIMD_FLOAT32_C(   130.14), EASYSIMD_FLOAT32_C(   293.52) },
      { EASYSIMD_FLOAT32_C(   -89.48), EASYSIMD_FLOAT32_C(  -934.97), EASYSIMD_FLOAT32_C(  -966.81), EASYSIMD_FLOAT32_C(   205.16),
        EASYSIMD_FLOAT32_C(   520.98), EASYSIMD_FLOAT32_C(  -615.17), EASYSIMD_FLOAT32_C(   -36.96), EASYSIMD_FLOAT32_C(    89.42),
        EASYSIMD_FLOAT32_C(   267.17), EASYSIMD_FLOAT32_C(   714.90), EASYSIMD_FLOAT32_C(   186.83), EASYSIMD_FLOAT32_C(  -109.23),
        EASYSIMD_FLOAT32_C(  -526.45), EASYSIMD_FLOAT32_C(   964.91), EASYSIMD_FLOAT32_C(   617.40), EASYSIMD_FLOAT32_C(   908.08) },
      {  INT32_C(  1692385033), -INT32_C(   992902210), -INT32_C(   173608878), -INT32_C(   639127479), -INT32_C(  1367543131),  INT32_C(  2116073808), -INT32_C(  1862700436),  INT32_C(   454408210),
         INT32_C(   897578103), -INT32_C(   889630600), -INT32_C(  1967153343), -INT32_C(   178018736), -INT32_C(   391913320),  INT32_C(    90686361),  INT32_C(  1217749046), -INT32_C(  1872516584) },
       INT32_C(         159),
      {     -EASYSIMD_MATH_INFINITYF,     -EASYSIMD_MATH_INFINITYF,      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(     1.57),
        EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(  -752.38), EASYSIMD_FLOAT32_C(    89.42),
                   EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    90.00), EASYSIMD_FLOAT32_C(     0.00),      EASYSIMD_MATH_INFINITYF,
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   683.59),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(    -1.00) } },
    { { EASYSIMD_FLOAT32_C(  -968.37), EASYSIMD_FLOAT32_C(   862.12), EASYSIMD_FLOAT32_C(   -32.24), EASYSIMD_FLOAT32_C(  -736.47),
        EASYSIMD_FLOAT32_C(  -397.51), EASYSIMD_FLOAT32_C(   215.37), EASYSIMD_FLOAT32_C(  -834.04), EASYSIMD_FLOAT32_C(    94.48),
        EASYSIMD_FLOAT32_C(  -134.80), EASYSIMD_FLOAT32_C(   397.86), EASYSIMD_FLOAT32_C(   814.81), EASYSIMD_FLOAT32_C(   171.35),
        EASYSIMD_FLOAT32_C(    81.45), EASYSIMD_FLOAT32_C(   -55.05), EASYSIMD_FLOAT32_C(  -535.13), EASYSIMD_FLOAT32_C(   991.98) },
      { EASYSIMD_FLOAT32_C(     9.98), EASYSIMD_FLOAT32_C(  -501.94), EASYSIMD_FLOAT32_C(   197.13), EASYSIMD_FLOAT32_C(  -469.04),
        EASYSIMD_FLOAT32_C(  -117.11), EASYSIMD_FLOAT32_C(  -839.83), EASYSIMD_FLOAT32_C(   620.38), EASYSIMD_FLOAT32_C(  -849.95),
        EASYSIMD_FLOAT32_C(   875.07), EASYSIMD_FLOAT32_C(  -192.79), EASYSIMD_FLOAT32_C(    40.82), EASYSIMD_FLOAT32_C(  -651.39),
        EASYSIMD_FLOAT32_C(  -227.88), EASYSIMD_FLOAT32_C(  -341.78), EASYSIMD_FLOAT32_C(  -743.31), EASYSIMD_FLOAT32_C(  -196.25) },
      {  INT32_C(  1960951603), -INT32_C(  1358978978), -INT32_C(   559717818), -INT32_C(   607762622),  INT32_C(  1088433418), -INT32_C(  1501006195), -INT32_C(  1086919648),  INT32_C(    47643599),
         INT32_C(   427276218), -INT32_C(  1647872425),  INT32_C(   461073368), -INT32_C(  1124711758), -INT32_C(    33695889),  INT32_C(  1839433037),  INT32_C(  1076746609), -INT32_C(  1891433516) },
       INT32_C(         187),
      { EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(     1.57), EASYSIMD_FLOAT32_C(     0.50),
        EASYSIMD_FLOAT32_C(  -397.51),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(     0.50),            EASYSIMD_MATH_NANF,
        EASYSIMD_FLOAT32_C(   875.07), EASYSIMD_FLOAT32_C(     1.57), EASYSIMD_FLOAT32_C(    40.82), EASYSIMD_FLOAT32_C(    90.00),
        EASYSIMD_FLOAT32_C(     1.57), EASYSIMD_FLOAT32_C(     1.57), EASYSIMD_FLOAT32_C(  -535.13), EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00) } },
    { { EASYSIMD_FLOAT32_C(  -479.66), EASYSIMD_FLOAT32_C(   224.45), EASYSIMD_FLOAT32_C(    67.28), EASYSIMD_FLOAT32_C(   122.84),
        EASYSIMD_FLOAT32_C(  -560.17), EASYSIMD_FLOAT32_C(   233.24), EASYSIMD_FLOAT32_C(  -782.68), EASYSIMD_FLOAT32_C(   305.03),
        EASYSIMD_FLOAT32_C(  -368.90), EASYSIMD_FLOAT32_C(  -967.87), EASYSIMD_FLOAT32_C(  -523.62), EASYSIMD_FLOAT32_C(   712.56),
        EASYSIMD_FLOAT32_C(   -22.92), EASYSIMD_FLOAT32_C(   -58.75), EASYSIMD_FLOAT32_C(   704.53), EASYSIMD_FLOAT32_C(   987.07) },
      { EASYSIMD_FLOAT32_C(   439.31), EASYSIMD_FLOAT32_C(   -98.34), EASYSIMD_FLOAT32_C(  -481.97), EASYSIMD_FLOAT32_C(  -677.80),
        EASYSIMD_FLOAT32_C(    61.83), EASYSIMD_FLOAT32_C(  -861.59), EASYSIMD_FLOAT32_C(  -527.75), EASYSIMD_FLOAT32_C(   -63.10),
        EASYSIMD_FLOAT32_C(   -54.38), EASYSIMD_FLOAT32_C(   513.07), EASYSIMD_FLOAT32_C(   285.51), EASYSIMD_FLOAT32_C(   717.74),
        EASYSIMD_FLOAT32_C(  -828.70), EASYSIMD_FLOAT32_C(   542.20), EASYSIMD_FLOAT32_C(  -478.51), EASYSIMD_FLOAT32_C(  -308.36) },
      {  INT32_C(   145780528), -INT32_C(  1943852070), -INT32_C(   582411667),  INT32_C(  1054492401),  INT32_C(  1034649035),  INT32_C(   712890454), -INT32_C(  1749434148), -INT32_C(  1431740038),
        -INT32_C(  1414309423), -INT32_C(   231156091), -INT32_C(   506494480), -INT32_C(  1843418681),  INT32_C(  2093992742),  INT32_C(  2141670819), -INT32_C(  2011799539), -INT32_C(  1825390398) },
       INT32_C(          25),
      { EASYSIMD_FLOAT32_C(  -479.66), EASYSIMD_FLOAT32_C(    90.00), EASYSIMD_FLOAT32_C(     1.57), EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00),
                   EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     1.00),
        EASYSIMD_FLOAT32_C(     0.50), EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(    -1.00),
        EASYSIMD_FLOAT32_C(    90.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00),            EASYSIMD_MATH_NANF } },
    { { EASYSIMD_FLOAT32_C(  -233.35), EASYSIMD_FLOAT32_C(   588.76), EASYSIMD_FLOAT32_C(   814.48), EASYSIMD_FLOAT32_C(   206.48),
        EASYSIMD_FLOAT32_C(  -178.00), EASYSIMD_FLOAT32_C(  -968.20), EASYSIMD_FLOAT32_C(  -488.50), EASYSIMD_FLOAT32_C(   453.11),
        EASYSIMD_FLOAT32_C(  -936.07), EASYSIMD_FLOAT32_C(   -12.11), EASYSIMD_FLOAT32_C(   165.66), EASYSIMD_FLOAT32_C(    41.01),
        EASYSIMD_FLOAT32_C(   929.14), EASYSIMD_FLOAT32_C(  -129.81), EASYSIMD_FLOAT32_C(    28.08), EASYSIMD_FLOAT32_C(   368.44) },
      { EASYSIMD_FLOAT32_C(   771.86), EASYSIMD_FLOAT32_C(   546.10), EASYSIMD_FLOAT32_C(   690.64), EASYSIMD_FLOAT32_C(  -166.31),
        EASYSIMD_FLOAT32_C(   684.51), EASYSIMD_FLOAT32_C(  -837.11), EASYSIMD_FLOAT32_C(   770.59), EASYSIMD_FLOAT32_C(  -369.87),
        EASYSIMD_FLOAT32_C(   675.97), EASYSIMD_FLOAT32_C(    56.10), EASYSIMD_FLOAT32_C(  -652.14), EASYSIMD_FLOAT32_C(   847.26),
        EASYSIMD_FLOAT32_C(  -401.70), EASYSIMD_FLOAT32_C(  -130.65), EASYSIMD_FLOAT32_C(  -461.10), EASYSIMD_FLOAT32_C(   364.95) },
      { -INT32_C(  1399753028), -INT32_C(  1097965321),  INT32_C(   810659082),  INT32_C(   481108088),  INT32_C(  2073777261),  INT32_C(  1979953844), -INT32_C(  1962330766), -INT32_C(   685094885),
        -INT32_C(  1232880706),  INT32_C(   628363547),  INT32_C(   928433599),  INT32_C(  1397949414),  INT32_C(   198111063),  INT32_C(   327340449),  INT32_C(   580815623), -INT32_C(  1829123885) },
       INT32_C(         131),
      { EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.50),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    90.00),
        EASYSIMD_FLOAT32_C(    -0.00),     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00),
        EASYSIMD_FLOAT32_C(     0.50),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    -0.00),      EASYSIMD_MATH_INFINITYF,
        EASYSIMD_FLOAT32_C(     0.50),            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    -1.00) } },
  };

  easysimd__m512 a, b, r;
  easysimd__m512i c;

  a = easysimd_mm512_loadu_ps(test_vec[0].a);
  b = easysimd_mm512_loadu_ps(test_vec[0].b);
  c = easysimd_mm512_loadu_epi32(test_vec[0].c);
  r = easysimd_mm512_fixupimm_ps(a, b, c, INT32_C(          60));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[1].a);
  b = easysimd_mm512_loadu_ps(test_vec[1].b);
  c = easysimd_mm512_loadu_epi32(test_vec[1].c);
  r = easysimd_mm512_fixupimm_ps(a, b, c, INT32_C(         143));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[2].a);
  b = easysimd_mm512_loadu_ps(test_vec[2].b);
  c = easysimd_mm512_loadu_epi32(test_vec[2].c);
  r = easysimd_mm512_fixupimm_ps(a, b, c, INT32_C(         133));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[3].a);
  b = easysimd_mm512_loadu_ps(test_vec[3].b);
  c = easysimd_mm512_loadu_epi32(test_vec[3].c);
  r = easysimd_mm512_fixupimm_ps(a, b, c, INT32_C(          36));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[4].a);
  b = easysimd_mm512_loadu_ps(test_vec[4].b);
  c = easysimd_mm512_loadu_epi32(test_vec[4].c);
  r = easysimd_mm512_fixupimm_ps(a, b, c, INT32_C(         159));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[5].a);
  b = easysimd_mm512_loadu_ps(test_vec[5].b);
  c = easysimd_mm512_loadu_epi32(test_vec[5].c);
  r = easysimd_mm512_fixupimm_ps(a, b, c, INT32_C(         187));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[6].a);
  b = easysimd_mm512_loadu_ps(test_vec[6].b);
  c = easysimd_mm512_loadu_epi32(test_vec[6].c);
  r = easysimd_mm512_fixupimm_ps(a, b, c, INT32_C(          25));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[7].a);
  b = easysimd_mm512_loadu_ps(test_vec[7].b);
  c = easysimd_mm512_loadu_epi32(test_vec[7].c);
  r = easysimd_mm512_fixupimm_ps(a, b, c, INT32_C(         131));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float32 values[8 * 2 * sizeof(easysimd__m512)];
  easysimd_test_x86_random_f32x16_full(8, 2, values, -1000.0f, 1000.0f, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_extract_f32x16(i, 2, 0, values);
    easysimd__m512 b = easysimd_test_x86_random_extract_f32x16(i, 2, 1, values);
    easysimd__m512i c = easysimd_test_x86_random_i32x16();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m512 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm512_fixupimm_ps, r, easysimd_mm512_setzero_ps(), imm8, a, b, c);

    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  fprintf(stderr, "-------------------------\n------------------------\n----------------------\n");
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_fixupimm_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[16];
    const easysimd__mmask16 k;
    const easysimd_float32 b[16];
    const int32_t c[16];
    const int imm8;
    const easysimd_float32 r[16];
  } test_vec[] = {
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    11.52),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   929.69),
        EASYSIMD_FLOAT32_C(  -164.22), EASYSIMD_FLOAT32_C(  -671.22), EASYSIMD_FLOAT32_C(  -338.31), EASYSIMD_FLOAT32_C(  -943.38),
        EASYSIMD_FLOAT32_C(  -204.60), EASYSIMD_FLOAT32_C(  -242.41), EASYSIMD_FLOAT32_C(  -153.78), EASYSIMD_FLOAT32_C(  -370.59),
        EASYSIMD_FLOAT32_C(  -911.93), EASYSIMD_FLOAT32_C(   185.86), EASYSIMD_FLOAT32_C(   977.68), EASYSIMD_FLOAT32_C(  -572.82) },
      UINT16_C(21745),
      { EASYSIMD_FLOAT32_C(   601.56),            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   151.49),
        EASYSIMD_FLOAT32_C(  -526.97), EASYSIMD_FLOAT32_C(   665.42), EASYSIMD_FLOAT32_C(  -228.74), EASYSIMD_FLOAT32_C(  -432.59),
        EASYSIMD_FLOAT32_C(  -861.10), EASYSIMD_FLOAT32_C(   937.48), EASYSIMD_FLOAT32_C(  -213.33), EASYSIMD_FLOAT32_C(    83.67),
        EASYSIMD_FLOAT32_C(   -69.75), EASYSIMD_FLOAT32_C(  -946.87), EASYSIMD_FLOAT32_C(   372.84), EASYSIMD_FLOAT32_C(   140.85) },
      {  INT32_C(   313384799),  INT32_C(  1572283312),  INT32_C(  1410692939), -INT32_C(   479587817), -INT32_C(   648402442), -INT32_C(   989262257),  INT32_C(  1641939429),  INT32_C(  1739968520),
         INT32_C(  1601790639), -INT32_C(   792972923), -INT32_C(  1071328600), -INT32_C(   610038044),  INT32_C(   129301943), -INT32_C(   238240500), -INT32_C(   950883649),  INT32_C(   690882426) },
       INT32_C(         105),
      { EASYSIMD_FLOAT32_C(   601.56), EASYSIMD_FLOAT32_C(    11.52),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   929.69),
        EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(    90.00), EASYSIMD_FLOAT32_C(  -228.74), EASYSIMD_FLOAT32_C(    -0.00),
        EASYSIMD_FLOAT32_C(  -204.60), EASYSIMD_FLOAT32_C(  -242.41), EASYSIMD_FLOAT32_C(  -153.78), EASYSIMD_FLOAT32_C(  -370.59),
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(   185.86), EASYSIMD_FLOAT32_C(    90.00), EASYSIMD_FLOAT32_C(  -572.82) } },
    { { EASYSIMD_FLOAT32_C(    64.65), EASYSIMD_FLOAT32_C(   971.39), EASYSIMD_FLOAT32_C(    70.55), EASYSIMD_FLOAT32_C(   900.43),
        EASYSIMD_FLOAT32_C(  -699.83), EASYSIMD_FLOAT32_C(   732.23), EASYSIMD_FLOAT32_C(   957.05), EASYSIMD_FLOAT32_C(    95.57),
        EASYSIMD_FLOAT32_C(  -510.17), EASYSIMD_FLOAT32_C(  -196.74), EASYSIMD_FLOAT32_C(   724.98), EASYSIMD_FLOAT32_C(  -422.10),
        EASYSIMD_FLOAT32_C(   989.12), EASYSIMD_FLOAT32_C(   702.66), EASYSIMD_FLOAT32_C(     5.08), EASYSIMD_FLOAT32_C(   590.68) },
      UINT16_C(17880),
      { EASYSIMD_FLOAT32_C(     6.22), EASYSIMD_FLOAT32_C(  -183.08), EASYSIMD_FLOAT32_C(  -257.83), EASYSIMD_FLOAT32_C(   479.25),
        EASYSIMD_FLOAT32_C(  -517.66), EASYSIMD_FLOAT32_C(   513.43), EASYSIMD_FLOAT32_C(  -953.35), EASYSIMD_FLOAT32_C(  -378.76),
        EASYSIMD_FLOAT32_C(   450.92), EASYSIMD_FLOAT32_C(  -166.68), EASYSIMD_FLOAT32_C(   704.91), EASYSIMD_FLOAT32_C(  -618.84),
        EASYSIMD_FLOAT32_C(  -113.55), EASYSIMD_FLOAT32_C(    77.75), EASYSIMD_FLOAT32_C(   522.02), EASYSIMD_FLOAT32_C(   951.10) },
      { -INT32_C(   484998722), -INT32_C(   462291903), -INT32_C(  1948177961), -INT32_C(    62264016),  INT32_C(   816187614), -INT32_C(    46718513), -INT32_C(   744185782),  INT32_C(  1293516174),
         INT32_C(  1060122878),  INT32_C(    52666668), -INT32_C(    91290422), -INT32_C(   789128974), -INT32_C(  1358848544),  INT32_C(   145504446),  INT32_C(  1742491865), -INT32_C(   877333043) },
       INT32_C(          37),
      { EASYSIMD_FLOAT32_C(    64.65), EASYSIMD_FLOAT32_C(   971.39), EASYSIMD_FLOAT32_C(    70.55), EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00),
        EASYSIMD_FLOAT32_C(  -699.83), EASYSIMD_FLOAT32_C(   732.23),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     1.57),
                   EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -196.74), EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(  -422.10),
        EASYSIMD_FLOAT32_C(   989.12), EASYSIMD_FLOAT32_C(   702.66),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   590.68) } },
    { { EASYSIMD_FLOAT32_C(    49.15), EASYSIMD_FLOAT32_C(  -407.44), EASYSIMD_FLOAT32_C(   851.52), EASYSIMD_FLOAT32_C(   349.32),
        EASYSIMD_FLOAT32_C(  -675.20), EASYSIMD_FLOAT32_C(   808.57), EASYSIMD_FLOAT32_C(  -555.11), EASYSIMD_FLOAT32_C(  -185.38),
        EASYSIMD_FLOAT32_C(  -388.17), EASYSIMD_FLOAT32_C(  -830.13), EASYSIMD_FLOAT32_C(   392.52), EASYSIMD_FLOAT32_C(  -399.05),
        EASYSIMD_FLOAT32_C(   872.53), EASYSIMD_FLOAT32_C(  -602.39), EASYSIMD_FLOAT32_C(  -808.37), EASYSIMD_FLOAT32_C(  -121.25) },
      UINT16_C(11654),
      { EASYSIMD_FLOAT32_C(   214.53), EASYSIMD_FLOAT32_C(   -66.21), EASYSIMD_FLOAT32_C(  -642.00), EASYSIMD_FLOAT32_C(   696.86),
        EASYSIMD_FLOAT32_C(  -552.78), EASYSIMD_FLOAT32_C(  -595.35), EASYSIMD_FLOAT32_C(  -681.90), EASYSIMD_FLOAT32_C(   898.14),
        EASYSIMD_FLOAT32_C(   237.97), EASYSIMD_FLOAT32_C(  -976.99), EASYSIMD_FLOAT32_C(  -720.70), EASYSIMD_FLOAT32_C(  -875.58),
        EASYSIMD_FLOAT32_C(   100.77), EASYSIMD_FLOAT32_C(   801.32), EASYSIMD_FLOAT32_C(  -924.49), EASYSIMD_FLOAT32_C(  -850.08) },
      { -INT32_C(   466464683),  INT32_C(  1102849099), -INT32_C(   169960204), -INT32_C(   147940277), -INT32_C(  2125985883),  INT32_C(   594941294), -INT32_C(   351822879),  INT32_C(  1125748205),
         INT32_C(   690441182),  INT32_C(  1667949679), -INT32_C(   866563712), -INT32_C(  1966897179), -INT32_C(    66385010),  INT32_C(    35619105),  INT32_C(   183314205), -INT32_C(  1840445772) },
       INT32_C(          81),
      { EASYSIMD_FLOAT32_C(    49.15), EASYSIMD_FLOAT32_C(   -66.21),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   349.32),
        EASYSIMD_FLOAT32_C(  -675.20), EASYSIMD_FLOAT32_C(   808.57), EASYSIMD_FLOAT32_C(  -555.11),     -EASYSIMD_MATH_INFINITYF,
                   EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -830.13), EASYSIMD_FLOAT32_C(    90.00), EASYSIMD_FLOAT32_C(     1.00),
        EASYSIMD_FLOAT32_C(   872.53), EASYSIMD_FLOAT32_C(  -602.39), EASYSIMD_FLOAT32_C(  -808.37), EASYSIMD_FLOAT32_C(  -121.25) } },
    { { EASYSIMD_FLOAT32_C(  -606.12), EASYSIMD_FLOAT32_C(   927.04), EASYSIMD_FLOAT32_C(   499.24), EASYSIMD_FLOAT32_C(  -281.33),
        EASYSIMD_FLOAT32_C(   735.61), EASYSIMD_FLOAT32_C(   944.13), EASYSIMD_FLOAT32_C(   533.30), EASYSIMD_FLOAT32_C(  -652.56),
        EASYSIMD_FLOAT32_C(  -886.00), EASYSIMD_FLOAT32_C(   -74.18), EASYSIMD_FLOAT32_C(   -51.61), EASYSIMD_FLOAT32_C(   986.53),
        EASYSIMD_FLOAT32_C(   323.43), EASYSIMD_FLOAT32_C(   140.01), EASYSIMD_FLOAT32_C(  -134.72), EASYSIMD_FLOAT32_C(  -462.04) },
      UINT16_C( 9817),
      { EASYSIMD_FLOAT32_C(  -926.19), EASYSIMD_FLOAT32_C(   223.28), EASYSIMD_FLOAT32_C(  -765.18), EASYSIMD_FLOAT32_C(  -478.97),
        EASYSIMD_FLOAT32_C(   627.93), EASYSIMD_FLOAT32_C(  -447.08), EASYSIMD_FLOAT32_C(  -580.83), EASYSIMD_FLOAT32_C(  -134.10),
        EASYSIMD_FLOAT32_C(  -424.06), EASYSIMD_FLOAT32_C(  -301.53), EASYSIMD_FLOAT32_C(    -9.68), EASYSIMD_FLOAT32_C(   676.71),
        EASYSIMD_FLOAT32_C(  -500.22), EASYSIMD_FLOAT32_C(    65.83), EASYSIMD_FLOAT32_C(   826.62), EASYSIMD_FLOAT32_C(  -106.34) },
      {  INT32_C(  2104482084),  INT32_C(  1761891493), -INT32_C(   361458977), -INT32_C(  1368615538), -INT32_C(  2049603177), -INT32_C(   510948973), -INT32_C(   682173156), -INT32_C(  1040339043),
         INT32_C(   775842952), -INT32_C(  1600699711),  INT32_C(  1669991380),  INT32_C(   940701345),  INT32_C(   331212415), -INT32_C(  1946924689), -INT32_C(  1016903130), -INT32_C(  2121965319) },
       INT32_C(         205),
      { EASYSIMD_FLOAT32_C(     1.57), EASYSIMD_FLOAT32_C(   927.04), EASYSIMD_FLOAT32_C(   499.24), EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   944.13), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(  -652.56),
        EASYSIMD_FLOAT32_C(  -886.00), EASYSIMD_FLOAT32_C(   -74.18),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   986.53),
        EASYSIMD_FLOAT32_C(   323.43), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -134.72), EASYSIMD_FLOAT32_C(  -462.04) } },
    { { EASYSIMD_FLOAT32_C(    -7.13), EASYSIMD_FLOAT32_C(   325.86), EASYSIMD_FLOAT32_C(   612.34), EASYSIMD_FLOAT32_C(  -271.52),
        EASYSIMD_FLOAT32_C(   269.99), EASYSIMD_FLOAT32_C(   145.64), EASYSIMD_FLOAT32_C(    75.91), EASYSIMD_FLOAT32_C(   383.99),
        EASYSIMD_FLOAT32_C(  -928.54), EASYSIMD_FLOAT32_C(  -975.70), EASYSIMD_FLOAT32_C(   370.52), EASYSIMD_FLOAT32_C(   394.89),
        EASYSIMD_FLOAT32_C(   164.31), EASYSIMD_FLOAT32_C(  -764.19), EASYSIMD_FLOAT32_C(   932.84), EASYSIMD_FLOAT32_C(   238.12) },
      UINT16_C(17926),
      { EASYSIMD_FLOAT32_C(   459.08), EASYSIMD_FLOAT32_C(  -832.34), EASYSIMD_FLOAT32_C(   759.15), EASYSIMD_FLOAT32_C(    87.01),
        EASYSIMD_FLOAT32_C(  -279.41), EASYSIMD_FLOAT32_C(  -821.69), EASYSIMD_FLOAT32_C(   952.91), EASYSIMD_FLOAT32_C(   296.52),
        EASYSIMD_FLOAT32_C(  -123.22), EASYSIMD_FLOAT32_C(   -56.77), EASYSIMD_FLOAT32_C(   -26.77), EASYSIMD_FLOAT32_C(   376.57),
        EASYSIMD_FLOAT32_C(  -990.93), EASYSIMD_FLOAT32_C(  -200.15), EASYSIMD_FLOAT32_C(  -729.77), EASYSIMD_FLOAT32_C(     1.94) },
      { -INT32_C(  1202529746),  INT32_C(  1320153917), -INT32_C(   382456277),  INT32_C(   892511297), -INT32_C(  1837148113), -INT32_C(  1594656741), -INT32_C(  1453014790),  INT32_C(  2096130638),
        -INT32_C(  2110504380),  INT32_C(  1624302389),  INT32_C(  1397294354), -INT32_C(   662078551), -INT32_C(   278198061),  INT32_C(  2072993409),  INT32_C(  1814360862), -INT32_C(  1561848739) },
       INT32_C(          86),
      { EASYSIMD_FLOAT32_C(    -7.13), EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(  -271.52),
        EASYSIMD_FLOAT32_C(   269.99), EASYSIMD_FLOAT32_C(   145.64), EASYSIMD_FLOAT32_C(    75.91), EASYSIMD_FLOAT32_C(   383.99),
        EASYSIMD_FLOAT32_C(  -928.54), EASYSIMD_FLOAT32_C(  -975.70),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   394.89),
        EASYSIMD_FLOAT32_C(   164.31), EASYSIMD_FLOAT32_C(  -764.19), EASYSIMD_FLOAT32_C(    90.00), EASYSIMD_FLOAT32_C(   238.12) } },
    { { EASYSIMD_FLOAT32_C(  -874.29), EASYSIMD_FLOAT32_C(   882.57), EASYSIMD_FLOAT32_C(   730.42), EASYSIMD_FLOAT32_C(   395.70),
        EASYSIMD_FLOAT32_C(    28.20), EASYSIMD_FLOAT32_C(  -193.67), EASYSIMD_FLOAT32_C(  -220.31), EASYSIMD_FLOAT32_C(    99.66),
        EASYSIMD_FLOAT32_C(  -169.37), EASYSIMD_FLOAT32_C(  -849.78), EASYSIMD_FLOAT32_C(  -505.45), EASYSIMD_FLOAT32_C(   994.95),
        EASYSIMD_FLOAT32_C(  -613.98), EASYSIMD_FLOAT32_C(  -572.60), EASYSIMD_FLOAT32_C(   233.07), EASYSIMD_FLOAT32_C(   845.11) },
      UINT16_C(62464),
      { EASYSIMD_FLOAT32_C(  -404.94), EASYSIMD_FLOAT32_C(    -7.79), EASYSIMD_FLOAT32_C(   -67.88), EASYSIMD_FLOAT32_C(   315.64),
        EASYSIMD_FLOAT32_C(   170.53), EASYSIMD_FLOAT32_C(  -114.97), EASYSIMD_FLOAT32_C(  -387.83), EASYSIMD_FLOAT32_C(  -952.69),
        EASYSIMD_FLOAT32_C(   828.26), EASYSIMD_FLOAT32_C(   585.39), EASYSIMD_FLOAT32_C(   423.88), EASYSIMD_FLOAT32_C(   837.33),
        EASYSIMD_FLOAT32_C(  -614.76), EASYSIMD_FLOAT32_C(   694.11), EASYSIMD_FLOAT32_C(  -160.73), EASYSIMD_FLOAT32_C(  -489.05) },
      {  INT32_C(   905515756), -INT32_C(   290348443), -INT32_C(   419986309),  INT32_C(    54818931),  INT32_C(   402023155), -INT32_C(  1238608690),  INT32_C(   450003702), -INT32_C(    99692018),
         INT32_C(  1244596452),  INT32_C(   624484522), -INT32_C(   670355611), -INT32_C(  1696903257),  INT32_C(  2142425777),  INT32_C(   473292326),  INT32_C(  1849034848), -INT32_C(  1083685670) },
       INT32_C(          76),
      { EASYSIMD_FLOAT32_C(  -874.29), EASYSIMD_FLOAT32_C(   882.57), EASYSIMD_FLOAT32_C(   730.42), EASYSIMD_FLOAT32_C(   395.70),
        EASYSIMD_FLOAT32_C(    28.20), EASYSIMD_FLOAT32_C(  -193.67), EASYSIMD_FLOAT32_C(  -220.31), EASYSIMD_FLOAT32_C(    99.66),
        EASYSIMD_FLOAT32_C(  -169.37), EASYSIMD_FLOAT32_C(  -849.78), EASYSIMD_FLOAT32_C(     1.57), EASYSIMD_FLOAT32_C(   994.95),
        EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(   694.11), EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00) } },
    { { EASYSIMD_FLOAT32_C(   576.67), EASYSIMD_FLOAT32_C(  -430.32), EASYSIMD_FLOAT32_C(   906.65), EASYSIMD_FLOAT32_C(  -395.12),
        EASYSIMD_FLOAT32_C(   376.01), EASYSIMD_FLOAT32_C(  -313.66), EASYSIMD_FLOAT32_C(   704.54), EASYSIMD_FLOAT32_C(  -793.36),
        EASYSIMD_FLOAT32_C(  -163.44), EASYSIMD_FLOAT32_C(  -800.91), EASYSIMD_FLOAT32_C(  -798.41), EASYSIMD_FLOAT32_C(   222.58),
        EASYSIMD_FLOAT32_C(  -373.52), EASYSIMD_FLOAT32_C(   434.65), EASYSIMD_FLOAT32_C(    67.68), EASYSIMD_FLOAT32_C(   221.54) },
      UINT16_C(16760),
      { EASYSIMD_FLOAT32_C(  -573.13), EASYSIMD_FLOAT32_C(   999.80), EASYSIMD_FLOAT32_C(  -462.82), EASYSIMD_FLOAT32_C(   597.40),
        EASYSIMD_FLOAT32_C(  -115.17), EASYSIMD_FLOAT32_C(   149.35), EASYSIMD_FLOAT32_C(   644.71), EASYSIMD_FLOAT32_C(  -286.90),
        EASYSIMD_FLOAT32_C(  -265.26), EASYSIMD_FLOAT32_C(    68.58), EASYSIMD_FLOAT32_C(  -449.57), EASYSIMD_FLOAT32_C(   119.98),
        EASYSIMD_FLOAT32_C(  -237.31), EASYSIMD_FLOAT32_C(   389.69), EASYSIMD_FLOAT32_C(   630.93), EASYSIMD_FLOAT32_C(  -660.64) },
      {  INT32_C(   678550812), -INT32_C(  1854465866),  INT32_C(  1700997555), -INT32_C(   565999192), -INT32_C(   605641819),  INT32_C(  2048966674), -INT32_C(  2012058497), -INT32_C(  2134209693),
         INT32_C(   514341736),  INT32_C(   112205651), -INT32_C(   244640952),  INT32_C(  1120906909),  INT32_C(  1679734098), -INT32_C(   169984395),  INT32_C(   243134890), -INT32_C(   460437636) },
       INT32_C(         130),
      { EASYSIMD_FLOAT32_C(   576.67), EASYSIMD_FLOAT32_C(  -430.32), EASYSIMD_FLOAT32_C(   906.65), EASYSIMD_FLOAT32_C(     1.57),
        EASYSIMD_FLOAT32_C(     0.50), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -793.36),
        EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(  -800.91), EASYSIMD_FLOAT32_C(  -798.41), EASYSIMD_FLOAT32_C(   222.58),
        EASYSIMD_FLOAT32_C(  -373.52), EASYSIMD_FLOAT32_C(   434.65), EASYSIMD_FLOAT32_C(    67.68), EASYSIMD_FLOAT32_C(   221.54) } },
    { { EASYSIMD_FLOAT32_C(   959.37), EASYSIMD_FLOAT32_C(   537.58), EASYSIMD_FLOAT32_C(   -55.76), EASYSIMD_FLOAT32_C(   335.39),
        EASYSIMD_FLOAT32_C(  -776.08), EASYSIMD_FLOAT32_C(  -351.23), EASYSIMD_FLOAT32_C(   542.03), EASYSIMD_FLOAT32_C(    60.48),
        EASYSIMD_FLOAT32_C(  -152.14), EASYSIMD_FLOAT32_C(   743.62), EASYSIMD_FLOAT32_C(  -716.94), EASYSIMD_FLOAT32_C(   474.34),
        EASYSIMD_FLOAT32_C(   178.27), EASYSIMD_FLOAT32_C(   350.74), EASYSIMD_FLOAT32_C(  -304.11), EASYSIMD_FLOAT32_C(   605.14) },
      UINT16_C(45909),
      { EASYSIMD_FLOAT32_C(   350.54), EASYSIMD_FLOAT32_C(   233.07), EASYSIMD_FLOAT32_C(   202.54), EASYSIMD_FLOAT32_C(  -764.62),
        EASYSIMD_FLOAT32_C(  -617.58), EASYSIMD_FLOAT32_C(  -152.76), EASYSIMD_FLOAT32_C(   -51.53), EASYSIMD_FLOAT32_C(   117.16),
        EASYSIMD_FLOAT32_C(   915.82), EASYSIMD_FLOAT32_C(   498.90), EASYSIMD_FLOAT32_C(  -762.86), EASYSIMD_FLOAT32_C(  -321.49),
        EASYSIMD_FLOAT32_C(  -111.41), EASYSIMD_FLOAT32_C(   868.08), EASYSIMD_FLOAT32_C(    17.87), EASYSIMD_FLOAT32_C(  -152.03) },
      {  INT32_C(  1220976348),  INT32_C(  1593205647), -INT32_C(  1005369178), -INT32_C(  1962768212), -INT32_C(    75715459),  INT32_C(  1212348602), -INT32_C(   545339940),  INT32_C(  2006111387),
         INT32_C(    29317490),  INT32_C(  1650439868), -INT32_C(  1423543554),  INT32_C(  2016815354), -INT32_C(  1888242987), -INT32_C(  2032618070),  INT32_C(   359028346), -INT32_C(  1668417494) },
       INT32_C(          81),
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   537.58), EASYSIMD_FLOAT32_C(    90.00), EASYSIMD_FLOAT32_C(   335.39),
        EASYSIMD_FLOAT32_C(     0.50), EASYSIMD_FLOAT32_C(  -351.23), EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(    60.48),
        EASYSIMD_FLOAT32_C(  -152.14),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -716.94), EASYSIMD_FLOAT32_C(   474.34),
        EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -304.11), EASYSIMD_FLOAT32_C(    90.00) } },
  };

  easysimd__m512 a, b, r;
  easysimd__m512i c;

  a = easysimd_mm512_loadu_ps(test_vec[0].a);
  b = easysimd_mm512_loadu_ps(test_vec[0].b);
  c = easysimd_mm512_loadu_epi32(test_vec[0].c);
  r = easysimd_mm512_mask_fixupimm_ps(a, test_vec[0].k, b, c, INT32_C(         105));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[1].a);
  b = easysimd_mm512_loadu_ps(test_vec[1].b);
  c = easysimd_mm512_loadu_epi32(test_vec[1].c);
  r = easysimd_mm512_mask_fixupimm_ps(a, test_vec[1].k, b, c, INT32_C(          37));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[2].a);
  b = easysimd_mm512_loadu_ps(test_vec[2].b);
  c = easysimd_mm512_loadu_epi32(test_vec[2].c);
  r = easysimd_mm512_mask_fixupimm_ps(a, test_vec[2].k, b, c, INT32_C(          81));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[3].a);
  b = easysimd_mm512_loadu_ps(test_vec[3].b);
  c = easysimd_mm512_loadu_epi32(test_vec[3].c);
  r = easysimd_mm512_mask_fixupimm_ps(a, test_vec[3].k, b, c, INT32_C(         205));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[4].a);
  b = easysimd_mm512_loadu_ps(test_vec[4].b);
  c = easysimd_mm512_loadu_epi32(test_vec[4].c);
  r = easysimd_mm512_mask_fixupimm_ps(a, test_vec[4].k, b, c, INT32_C(          86));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[5].a);
  b = easysimd_mm512_loadu_ps(test_vec[5].b);
  c = easysimd_mm512_loadu_epi32(test_vec[5].c);
  r = easysimd_mm512_mask_fixupimm_ps(a, test_vec[5].k, b, c, INT32_C(          76));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[6].a);
  b = easysimd_mm512_loadu_ps(test_vec[6].b);
  c = easysimd_mm512_loadu_epi32(test_vec[6].c);
  r = easysimd_mm512_mask_fixupimm_ps(a, test_vec[6].k, b, c, INT32_C(         130));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[7].a);
  b = easysimd_mm512_loadu_ps(test_vec[7].b);
  c = easysimd_mm512_loadu_epi32(test_vec[7].c);
  r = easysimd_mm512_mask_fixupimm_ps(a, test_vec[7].k, b, c, INT32_C(          81));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float32 values[8 * 2 * sizeof(easysimd__m512)];
  easysimd_test_x86_random_f32x16_full(8, 2, values, -1000.0f, 1000.0f, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_extract_f32x16(i, 2, 0, values);
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512 b = easysimd_test_x86_random_extract_f32x16(i, 2, 1, values);
    easysimd__m512i c = easysimd_test_x86_random_i32x16();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m512 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm512_mask_fixupimm_ps, r, easysimd_mm512_setzero_ps(), imm8, a, k, b, c);

    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  fprintf(stderr, "-------------------------\n------------------------\n----------------------\n");
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_fixupimm_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k;
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const int32_t c[16];
    const int imm8;
    const easysimd_float32 r[16];
  } test_vec[] = {
    { UINT16_C( 4530),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -915.59),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -611.87),
        EASYSIMD_FLOAT32_C(   972.93), EASYSIMD_FLOAT32_C(  -368.90), EASYSIMD_FLOAT32_C(   180.66), EASYSIMD_FLOAT32_C(   295.65),
        EASYSIMD_FLOAT32_C(  -338.99), EASYSIMD_FLOAT32_C(  -384.12), EASYSIMD_FLOAT32_C(  -256.69), EASYSIMD_FLOAT32_C(  -116.90),
        EASYSIMD_FLOAT32_C(   336.00), EASYSIMD_FLOAT32_C(   767.24), EASYSIMD_FLOAT32_C(   339.49), EASYSIMD_FLOAT32_C(  -776.36) },
      { EASYSIMD_FLOAT32_C(   426.32),            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   200.37),
        EASYSIMD_FLOAT32_C(   186.37), EASYSIMD_FLOAT32_C(   203.64), EASYSIMD_FLOAT32_C(  -405.80), EASYSIMD_FLOAT32_C(   404.64),
        EASYSIMD_FLOAT32_C(   860.07), EASYSIMD_FLOAT32_C(   721.39), EASYSIMD_FLOAT32_C(  -837.45), EASYSIMD_FLOAT32_C(    -5.34),
        EASYSIMD_FLOAT32_C(   413.01), EASYSIMD_FLOAT32_C(  -154.92), EASYSIMD_FLOAT32_C(  -941.34), EASYSIMD_FLOAT32_C(  -275.03) },
      { -INT32_C(   156968499),  INT32_C(   704587896), -INT32_C(  2049974577),  INT32_C(   866252229), -INT32_C(   574602056),  INT32_C(  1633063746),  INT32_C(  2001845234),  INT32_C(  1971848104),
         INT32_C(  1416310236),  INT32_C(   612264533),  INT32_C(   246041929), -INT32_C(    79541438), -INT32_C(   774372721), -INT32_C(  2009976938), -INT32_C(  1828748310),  INT32_C(  1661438087) },
       INT32_C(         181),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     1.57),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -0.00),
             EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     1.57), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(14045),
      { EASYSIMD_FLOAT32_C(   -70.51), EASYSIMD_FLOAT32_C(  -358.94), EASYSIMD_FLOAT32_C(   113.10), EASYSIMD_FLOAT32_C(   -97.59),
        EASYSIMD_FLOAT32_C(   272.16), EASYSIMD_FLOAT32_C(  -706.25), EASYSIMD_FLOAT32_C(  -801.94), EASYSIMD_FLOAT32_C(   933.18),
        EASYSIMD_FLOAT32_C(   -90.36), EASYSIMD_FLOAT32_C(   -58.63), EASYSIMD_FLOAT32_C(  -183.72), EASYSIMD_FLOAT32_C(  -754.36),
        EASYSIMD_FLOAT32_C(  -291.39), EASYSIMD_FLOAT32_C(  -844.23), EASYSIMD_FLOAT32_C(  -530.72), EASYSIMD_FLOAT32_C(  -865.07) },
      { EASYSIMD_FLOAT32_C(   -64.89), EASYSIMD_FLOAT32_C(  -143.98), EASYSIMD_FLOAT32_C(   335.30), EASYSIMD_FLOAT32_C(  -878.52),
        EASYSIMD_FLOAT32_C(  -940.34), EASYSIMD_FLOAT32_C(   929.50), EASYSIMD_FLOAT32_C(   526.12), EASYSIMD_FLOAT32_C(   919.74),
        EASYSIMD_FLOAT32_C(   650.89), EASYSIMD_FLOAT32_C(   688.67), EASYSIMD_FLOAT32_C(   -85.61), EASYSIMD_FLOAT32_C(    63.90),
        EASYSIMD_FLOAT32_C(  -466.25), EASYSIMD_FLOAT32_C(   -26.95), EASYSIMD_FLOAT32_C(   788.87), EASYSIMD_FLOAT32_C(   463.23) },
      { -INT32_C(   662493650),  INT32_C(  1998833205), -INT32_C(  1720077631), -INT32_C(  1245180029), -INT32_C(  1757826409),  INT32_C(  1277149252), -INT32_C(   608185309),  INT32_C(   202480862),
        -INT32_C(   119171645),  INT32_C(   477038683),  INT32_C(  1052108987), -INT32_C(  1879802120),  INT32_C(  1965436208),  INT32_C(   281101805), -INT32_C(   135495655), -INT32_C(   553321188) },
       INT32_C(         146),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00),      EASYSIMD_MATH_INFINITYF,
        EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.57), EASYSIMD_FLOAT32_C(   933.18),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   688.67), EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(     0.00),
             EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(  -844.23), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(18161),
      { EASYSIMD_FLOAT32_C(   614.11), EASYSIMD_FLOAT32_C(   -98.04), EASYSIMD_FLOAT32_C(  -634.35), EASYSIMD_FLOAT32_C(  -113.73),
        EASYSIMD_FLOAT32_C(   195.71), EASYSIMD_FLOAT32_C(  -436.29), EASYSIMD_FLOAT32_C(  -180.56), EASYSIMD_FLOAT32_C(  -894.65),
        EASYSIMD_FLOAT32_C(   505.08), EASYSIMD_FLOAT32_C(   635.72), EASYSIMD_FLOAT32_C(  -649.01), EASYSIMD_FLOAT32_C(  -786.31),
        EASYSIMD_FLOAT32_C(   791.49), EASYSIMD_FLOAT32_C(  -179.73), EASYSIMD_FLOAT32_C(  -651.38), EASYSIMD_FLOAT32_C(  -273.39) },
      { EASYSIMD_FLOAT32_C(   676.29), EASYSIMD_FLOAT32_C(   683.92), EASYSIMD_FLOAT32_C(  -151.91), EASYSIMD_FLOAT32_C(   735.95),
        EASYSIMD_FLOAT32_C(   613.41), EASYSIMD_FLOAT32_C(  -625.79), EASYSIMD_FLOAT32_C(   655.69), EASYSIMD_FLOAT32_C(   264.30),
        EASYSIMD_FLOAT32_C(  -937.12), EASYSIMD_FLOAT32_C(  -429.92), EASYSIMD_FLOAT32_C(  -671.80), EASYSIMD_FLOAT32_C(  -403.37),
        EASYSIMD_FLOAT32_C(   543.13), EASYSIMD_FLOAT32_C(  -882.94), EASYSIMD_FLOAT32_C(  -940.14), EASYSIMD_FLOAT32_C(   157.24) },
      { -INT32_C(  1087394807), -INT32_C(   549640213), -INT32_C(   586388042), -INT32_C(  1557988894), -INT32_C(   182240247),  INT32_C(   938688563), -INT32_C(   148863713),  INT32_C(  2084377203),
        -INT32_C(  1455723330),  INT32_C(  1250457747), -INT32_C(   936930074), -INT32_C(  1754510963), -INT32_C(  1181970555), -INT32_C(   269451313),  INT32_C(  2028343557), -INT32_C(   504093917) },
       INT32_C(         144),
      { EASYSIMD_FLOAT32_C(     0.50), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(    -0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C( 5032),
      { EASYSIMD_FLOAT32_C(    19.02), EASYSIMD_FLOAT32_C(  -574.49), EASYSIMD_FLOAT32_C(  -956.50), EASYSIMD_FLOAT32_C(  -785.26),
        EASYSIMD_FLOAT32_C(   -10.78), EASYSIMD_FLOAT32_C(  -137.05), EASYSIMD_FLOAT32_C(  -679.91), EASYSIMD_FLOAT32_C(  -505.71),
        EASYSIMD_FLOAT32_C(  -501.33), EASYSIMD_FLOAT32_C(  -328.92), EASYSIMD_FLOAT32_C(  -292.02), EASYSIMD_FLOAT32_C(  -709.84),
        EASYSIMD_FLOAT32_C(   491.34), EASYSIMD_FLOAT32_C(    56.60), EASYSIMD_FLOAT32_C(    16.77), EASYSIMD_FLOAT32_C(   167.63) },
      { EASYSIMD_FLOAT32_C(  -259.48), EASYSIMD_FLOAT32_C(   864.86), EASYSIMD_FLOAT32_C(   -96.41), EASYSIMD_FLOAT32_C(  -646.07),
        EASYSIMD_FLOAT32_C(  -760.93), EASYSIMD_FLOAT32_C(  -440.72), EASYSIMD_FLOAT32_C(   618.23), EASYSIMD_FLOAT32_C(  -698.05),
        EASYSIMD_FLOAT32_C(   129.36), EASYSIMD_FLOAT32_C(   946.42), EASYSIMD_FLOAT32_C(  -101.42), EASYSIMD_FLOAT32_C(  -327.51),
        EASYSIMD_FLOAT32_C(  -936.52), EASYSIMD_FLOAT32_C(   -41.56), EASYSIMD_FLOAT32_C(   829.73), EASYSIMD_FLOAT32_C(    82.51) },
      { -INT32_C(  1800892819), -INT32_C(  1008847529),  INT32_C(  1498571724),  INT32_C(   232268316), -INT32_C(   148972271),  INT32_C(  1243234645), -INT32_C(  1384469982),  INT32_C(  1002513102),
         INT32_C(   147876273),  INT32_C(  1808510622),  INT32_C(   784604433),  INT32_C(  1346083903), -INT32_C(   817407622), -INT32_C(  1139187046), -INT32_C(   630549748),  INT32_C(  1729506230) },
       INT32_C(         148),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.57),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.50),
        EASYSIMD_FLOAT32_C(  -501.33),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(14996),
      { EASYSIMD_FLOAT32_C(   383.95), EASYSIMD_FLOAT32_C(   873.23), EASYSIMD_FLOAT32_C(   297.25), EASYSIMD_FLOAT32_C(  -626.83),
        EASYSIMD_FLOAT32_C(  -263.82), EASYSIMD_FLOAT32_C(   617.33), EASYSIMD_FLOAT32_C(  -132.54), EASYSIMD_FLOAT32_C(   234.85),
        EASYSIMD_FLOAT32_C(  -711.59), EASYSIMD_FLOAT32_C(   575.45), EASYSIMD_FLOAT32_C(   525.01), EASYSIMD_FLOAT32_C(   779.75),
        EASYSIMD_FLOAT32_C(  -367.96), EASYSIMD_FLOAT32_C(  -458.22), EASYSIMD_FLOAT32_C(   -52.61), EASYSIMD_FLOAT32_C(   372.56) },
      { EASYSIMD_FLOAT32_C(  -593.37), EASYSIMD_FLOAT32_C(   850.97), EASYSIMD_FLOAT32_C(   726.49), EASYSIMD_FLOAT32_C(  -354.30),
        EASYSIMD_FLOAT32_C(  -589.75), EASYSIMD_FLOAT32_C(   344.72), EASYSIMD_FLOAT32_C(   -52.35), EASYSIMD_FLOAT32_C(   539.61),
        EASYSIMD_FLOAT32_C(   291.14), EASYSIMD_FLOAT32_C(   846.23), EASYSIMD_FLOAT32_C(  -787.90), EASYSIMD_FLOAT32_C(   354.62),
        EASYSIMD_FLOAT32_C(  -195.33), EASYSIMD_FLOAT32_C(  -958.17), EASYSIMD_FLOAT32_C(  -562.87), EASYSIMD_FLOAT32_C(  -811.37) },
      {  INT32_C(  1633133981),  INT32_C(   251566035),  INT32_C(   795310311),  INT32_C(  1636892999),  INT32_C(   687119806), -INT32_C(  1907119496), -INT32_C(  2139822319), -INT32_C(  1212544999),
        -INT32_C(  2129128787), -INT32_C(  1869670743), -INT32_C(   692062322), -INT32_C(  1237888776),  INT32_C(  1725836270), -INT32_C(   403426858),  INT32_C(   845703192),  INT32_C(   501817968) },
       INT32_C(          52),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.50),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    -1.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.50),
            -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(11800),
      { EASYSIMD_FLOAT32_C(   915.06), EASYSIMD_FLOAT32_C(   734.38), EASYSIMD_FLOAT32_C(  -438.20), EASYSIMD_FLOAT32_C(  -348.76),
        EASYSIMD_FLOAT32_C(   351.71), EASYSIMD_FLOAT32_C(   429.26), EASYSIMD_FLOAT32_C(   886.09), EASYSIMD_FLOAT32_C(   640.12),
        EASYSIMD_FLOAT32_C(     4.71), EASYSIMD_FLOAT32_C(   411.10), EASYSIMD_FLOAT32_C(   419.87), EASYSIMD_FLOAT32_C(   636.75),
        EASYSIMD_FLOAT32_C(   952.88), EASYSIMD_FLOAT32_C(  -632.74), EASYSIMD_FLOAT32_C(     9.31), EASYSIMD_FLOAT32_C(  -640.49) },
      { EASYSIMD_FLOAT32_C(  -781.77), EASYSIMD_FLOAT32_C(  -264.20), EASYSIMD_FLOAT32_C(     5.21), EASYSIMD_FLOAT32_C(  -371.51),
        EASYSIMD_FLOAT32_C(  -919.48), EASYSIMD_FLOAT32_C(   952.86), EASYSIMD_FLOAT32_C(  -831.90), EASYSIMD_FLOAT32_C(   371.66),
        EASYSIMD_FLOAT32_C(   799.09), EASYSIMD_FLOAT32_C(  -619.80), EASYSIMD_FLOAT32_C(  -273.72), EASYSIMD_FLOAT32_C(  -396.24),
        EASYSIMD_FLOAT32_C(  -577.97), EASYSIMD_FLOAT32_C(   163.41), EASYSIMD_FLOAT32_C(  -207.61), EASYSIMD_FLOAT32_C(  -662.91) },
      {  INT32_C(   757442158), -INT32_C(  1283580548), -INT32_C(  1293980460), -INT32_C(   958089774),  INT32_C(    70188188),  INT32_C(   388472366),  INT32_C(  1511611323), -INT32_C(  1484246727),
         INT32_C(  1423224279), -INT32_C(  1643687222), -INT32_C(  1873680706),  INT32_C(   945173915),  INT32_C(  1614578737), -INT32_C(   562601182), -INT32_C(   130510657), -INT32_C(  1717583679) },
       INT32_C(         109),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),     -EASYSIMD_MATH_INFINITYF,
            -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(   419.87), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.57), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(62662),
      { EASYSIMD_FLOAT32_C(  -102.21), EASYSIMD_FLOAT32_C(   354.19), EASYSIMD_FLOAT32_C(   -11.66), EASYSIMD_FLOAT32_C(  -750.50),
        EASYSIMD_FLOAT32_C(  -216.55), EASYSIMD_FLOAT32_C(  -125.57), EASYSIMD_FLOAT32_C(   889.62), EASYSIMD_FLOAT32_C(   788.16),
        EASYSIMD_FLOAT32_C(  -714.47), EASYSIMD_FLOAT32_C(   309.50), EASYSIMD_FLOAT32_C(   424.91), EASYSIMD_FLOAT32_C(  -761.59),
        EASYSIMD_FLOAT32_C(   676.76), EASYSIMD_FLOAT32_C(  -565.77), EASYSIMD_FLOAT32_C(  -402.08), EASYSIMD_FLOAT32_C(   894.99) },
      { EASYSIMD_FLOAT32_C(   170.03), EASYSIMD_FLOAT32_C(   603.13), EASYSIMD_FLOAT32_C(  -476.52), EASYSIMD_FLOAT32_C(   250.55),
        EASYSIMD_FLOAT32_C(   555.99), EASYSIMD_FLOAT32_C(  -308.42), EASYSIMD_FLOAT32_C(  -377.79), EASYSIMD_FLOAT32_C(   355.08),
        EASYSIMD_FLOAT32_C(    71.78), EASYSIMD_FLOAT32_C(   348.49), EASYSIMD_FLOAT32_C(   958.85), EASYSIMD_FLOAT32_C(   493.81),
        EASYSIMD_FLOAT32_C(  -488.10), EASYSIMD_FLOAT32_C(  -248.76), EASYSIMD_FLOAT32_C(   830.90), EASYSIMD_FLOAT32_C(   409.69) },
      {  INT32_C(   668697814),  INT32_C(  1801221653), -INT32_C(   336556626),  INT32_C(  1699615469),  INT32_C(   687148528), -INT32_C(  1528252667),  INT32_C(  1025004880),  INT32_C(  1664212621),
         INT32_C(  2005535842),  INT32_C(   837019267),  INT32_C(  1629279091), -INT32_C(   691639323),  INT32_C(  2130623352), -INT32_C(  1037899918), -INT32_C(   906020292),  INT32_C(  2066493720) },
       INT32_C(          61),
      { EASYSIMD_FLOAT32_C(     0.00),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(     0.50), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     1.57),      EASYSIMD_MATH_INFINITYF,
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(340282346638528859811704183484516925440.00),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    90.00), EASYSIMD_FLOAT32_C(    -0.00) } },
    { UINT16_C(54684),
      { EASYSIMD_FLOAT32_C(  -894.58), EASYSIMD_FLOAT32_C(  -180.76), EASYSIMD_FLOAT32_C(   659.19), EASYSIMD_FLOAT32_C(  -111.12),
        EASYSIMD_FLOAT32_C(   693.67), EASYSIMD_FLOAT32_C(   548.82), EASYSIMD_FLOAT32_C(  -322.96), EASYSIMD_FLOAT32_C(   979.20),
        EASYSIMD_FLOAT32_C(  -141.69), EASYSIMD_FLOAT32_C(  -898.05), EASYSIMD_FLOAT32_C(  -782.40), EASYSIMD_FLOAT32_C(  -464.93),
        EASYSIMD_FLOAT32_C(  -463.82), EASYSIMD_FLOAT32_C(  -184.48), EASYSIMD_FLOAT32_C(  -569.93), EASYSIMD_FLOAT32_C(   706.20) },
      { EASYSIMD_FLOAT32_C(  -581.35), EASYSIMD_FLOAT32_C(   -46.46), EASYSIMD_FLOAT32_C(   -43.25), EASYSIMD_FLOAT32_C(   974.64),
        EASYSIMD_FLOAT32_C(   645.12), EASYSIMD_FLOAT32_C(   578.96), EASYSIMD_FLOAT32_C(   329.73), EASYSIMD_FLOAT32_C(  -283.11),
        EASYSIMD_FLOAT32_C(   -72.55), EASYSIMD_FLOAT32_C(   288.57), EASYSIMD_FLOAT32_C(  -789.30), EASYSIMD_FLOAT32_C(   439.35),
        EASYSIMD_FLOAT32_C(  -960.19), EASYSIMD_FLOAT32_C(  -958.40), EASYSIMD_FLOAT32_C(  -150.96), EASYSIMD_FLOAT32_C(  -854.77) },
      {  INT32_C(   245895410),  INT32_C(   930713201),  INT32_C(  1660088932), -INT32_C(  1840683664),  INT32_C(   667780647),  INT32_C(  2086200655),  INT32_C(  1395823968),  INT32_C(  1210634070),
         INT32_C(  1347867103), -INT32_C(  1014509473),  INT32_C(   841316802),  INT32_C(   113536990),  INT32_C(  1143837173), -INT32_C(   675248777),  INT32_C(  1881862938), -INT32_C(  1581755454) },
       INT32_C(          36),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    -1.00),
                   EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  -141.69), EASYSIMD_FLOAT32_C(     0.00),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(     0.00),
            -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -569.93), EASYSIMD_FLOAT32_C(  -854.77) } },
  };

  easysimd__m512 a, b, r;
  easysimd__m512i c;

  a = easysimd_mm512_loadu_ps(test_vec[0].a);
  b = easysimd_mm512_loadu_ps(test_vec[0].b);
  c = easysimd_mm512_loadu_epi32(test_vec[0].c);
  r = easysimd_mm512_maskz_fixupimm_ps(test_vec[0].k, a, b, c, INT32_C(         181));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[1].a);
  b = easysimd_mm512_loadu_ps(test_vec[1].b);
  c = easysimd_mm512_loadu_epi32(test_vec[1].c);
  r = easysimd_mm512_maskz_fixupimm_ps(test_vec[1].k, a, b, c, INT32_C(         146));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[2].a);
  b = easysimd_mm512_loadu_ps(test_vec[2].b);
  c = easysimd_mm512_loadu_epi32(test_vec[2].c);
  r = easysimd_mm512_maskz_fixupimm_ps(test_vec[2].k, a, b, c, INT32_C(         144));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[3].a);
  b = easysimd_mm512_loadu_ps(test_vec[3].b);
  c = easysimd_mm512_loadu_epi32(test_vec[3].c);
  r = easysimd_mm512_maskz_fixupimm_ps(test_vec[3].k, a, b, c, INT32_C(         148));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[4].a);
  b = easysimd_mm512_loadu_ps(test_vec[4].b);
  c = easysimd_mm512_loadu_epi32(test_vec[4].c);
  r = easysimd_mm512_maskz_fixupimm_ps(test_vec[4].k, a, b, c, INT32_C(          52));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[5].a);
  b = easysimd_mm512_loadu_ps(test_vec[5].b);
  c = easysimd_mm512_loadu_epi32(test_vec[5].c);
  r = easysimd_mm512_maskz_fixupimm_ps(test_vec[5].k, a, b, c, INT32_C(         109));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[6].a);
  b = easysimd_mm512_loadu_ps(test_vec[6].b);
  c = easysimd_mm512_loadu_epi32(test_vec[6].c);
  r = easysimd_mm512_maskz_fixupimm_ps(test_vec[6].k, a, b, c, INT32_C(          61));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[7].a);
  b = easysimd_mm512_loadu_ps(test_vec[7].b);
  c = easysimd_mm512_loadu_epi32(test_vec[7].c);
  r = easysimd_mm512_maskz_fixupimm_ps(test_vec[7].k, a, b, c, INT32_C(          36));
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float32 values[8 * 2 * sizeof(easysimd__m512)];
  easysimd_test_x86_random_f32x16_full(8, 2, values, -1000.0f, 1000.0f, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512 a = easysimd_test_x86_random_extract_f32x16(i, 2, 0, values);
    easysimd__m512 b = easysimd_test_x86_random_extract_f32x16(i, 2, 1, values);
    easysimd__m512i c = easysimd_test_x86_random_i32x16();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m512 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm512_maskz_fixupimm_ps, r, easysimd_mm512_setzero_ps(), imm8, k, a, b, c);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  fprintf(stderr, "-------------------------\n------------------------\n----------------------\n");
  return 1;
#endif
}

static int
test_easysimd_mm_fixupimm_ss (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const int32_t c[4];
    const int imm8;
    const easysimd_float32 r[4];
  } test_vec[] = {
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -551.70),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   860.02) },
      { EASYSIMD_FLOAT32_C(  -735.62),            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   790.88) },
      { -INT32_C(  1265195033),  INT32_C(   140789386), -INT32_C(  1899312875), -INT32_C(  1770193656) },
       INT32_C(         221),
      {     -EASYSIMD_MATH_INFINITYF,            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   790.88) } },
    { { EASYSIMD_FLOAT32_C(  -566.66), EASYSIMD_FLOAT32_C(  -571.40), EASYSIMD_FLOAT32_C(  -782.02), EASYSIMD_FLOAT32_C(   251.94) },
      { EASYSIMD_FLOAT32_C(   742.03), EASYSIMD_FLOAT32_C(  -460.96), EASYSIMD_FLOAT32_C(  -135.23), EASYSIMD_FLOAT32_C(  -190.10) },
      { -INT32_C(  1353407648),  INT32_C(  1058776384), -INT32_C(  1024844325),  INT32_C(  1316389060) },
       INT32_C(         198),
      { EASYSIMD_FLOAT32_C(     1.00), EASYSIMD_FLOAT32_C(  -460.96), EASYSIMD_FLOAT32_C(  -135.23), EASYSIMD_FLOAT32_C(  -190.10) } },
    { { EASYSIMD_FLOAT32_C(   660.22), EASYSIMD_FLOAT32_C(  -229.57), EASYSIMD_FLOAT32_C(    69.62), EASYSIMD_FLOAT32_C(   482.86) },
      { EASYSIMD_FLOAT32_C(  -269.44), EASYSIMD_FLOAT32_C(   498.43), EASYSIMD_FLOAT32_C(   544.01), EASYSIMD_FLOAT32_C(  -439.37) },
      { -INT32_C(  1385619292), -INT32_C(    79436514), -INT32_C(   752182669),  INT32_C(  2122481213) },
       INT32_C(          36),
      { EASYSIMD_FLOAT32_C(     1.57), EASYSIMD_FLOAT32_C(   498.43), EASYSIMD_FLOAT32_C(   544.01), EASYSIMD_FLOAT32_C(  -439.37) } },
    { { EASYSIMD_FLOAT32_C(  -597.94), EASYSIMD_FLOAT32_C(  -379.85), EASYSIMD_FLOAT32_C(   -27.17), EASYSIMD_FLOAT32_C(    48.01) },
      { EASYSIMD_FLOAT32_C(  -137.68), EASYSIMD_FLOAT32_C(  -230.12), EASYSIMD_FLOAT32_C(   577.75), EASYSIMD_FLOAT32_C(  -764.05) },
      {  INT32_C(  2143398075), -INT32_C(   271763672), -INT32_C(  1211489262),  INT32_C(  1650734148) },
       INT32_C(          25),
      { EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(  -230.12), EASYSIMD_FLOAT32_C(   577.75), EASYSIMD_FLOAT32_C(  -764.05) } },
    { { EASYSIMD_FLOAT32_C(   218.17), EASYSIMD_FLOAT32_C(   867.81), EASYSIMD_FLOAT32_C(  -904.03), EASYSIMD_FLOAT32_C(   482.55) },
      { EASYSIMD_FLOAT32_C(   374.21), EASYSIMD_FLOAT32_C(  -537.12), EASYSIMD_FLOAT32_C(   273.43), EASYSIMD_FLOAT32_C(   807.55) },
      {  INT32_C(  2120255553),  INT32_C(   737993479),  INT32_C(  1009433217), -INT32_C(  1967395998) },
       INT32_C(          34),
      { EASYSIMD_FLOAT32_C(    -0.00), EASYSIMD_FLOAT32_C(  -537.12), EASYSIMD_FLOAT32_C(   273.43), EASYSIMD_FLOAT32_C(   807.55) } },
    { { EASYSIMD_FLOAT32_C(  -108.52), EASYSIMD_FLOAT32_C(   491.41), EASYSIMD_FLOAT32_C(    59.49), EASYSIMD_FLOAT32_C(  -366.49) },
      { EASYSIMD_FLOAT32_C(  -969.54), EASYSIMD_FLOAT32_C(   924.26), EASYSIMD_FLOAT32_C(   443.40), EASYSIMD_FLOAT32_C(   690.68) },
      { -INT32_C(   236174164), -INT32_C(  1856810888),  INT32_C(   941535735),  INT32_C(  1102479162) },
       INT32_C(          98),
      { EASYSIMD_FLOAT32_C(  -969.54), EASYSIMD_FLOAT32_C(   924.26), EASYSIMD_FLOAT32_C(   443.40), EASYSIMD_FLOAT32_C(   690.68) } },
    { { EASYSIMD_FLOAT32_C(  -305.31), EASYSIMD_FLOAT32_C(  -486.97), EASYSIMD_FLOAT32_C(   173.54), EASYSIMD_FLOAT32_C(   425.25) },
      { EASYSIMD_FLOAT32_C(  -988.54), EASYSIMD_FLOAT32_C(  -282.45), EASYSIMD_FLOAT32_C(   985.88), EASYSIMD_FLOAT32_C(  -586.48) },
      { -INT32_C(   820013459), -INT32_C(  1554392447),  INT32_C(   265868130), -INT32_C(  1895775209) },
       INT32_C(          20),
      { EASYSIMD_FLOAT32_C(-340282346638528859811704183484516925440.00), EASYSIMD_FLOAT32_C(  -282.45), EASYSIMD_FLOAT32_C(   985.88), EASYSIMD_FLOAT32_C(  -586.48) } },
    { { EASYSIMD_FLOAT32_C(   337.70), EASYSIMD_FLOAT32_C(   -41.29), EASYSIMD_FLOAT32_C(   461.53), EASYSIMD_FLOAT32_C(  -799.98) },
      { EASYSIMD_FLOAT32_C(   728.59), EASYSIMD_FLOAT32_C(    39.27), EASYSIMD_FLOAT32_C(  -564.03), EASYSIMD_FLOAT32_C(   -53.24) },
      {  INT32_C(  1061371653),  INT32_C(   545323710),  INT32_C(   436464813),  INT32_C(    65610370) },
       INT32_C(         252),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    39.27), EASYSIMD_FLOAT32_C(  -564.03), EASYSIMD_FLOAT32_C(   -53.24) } },
  };

  easysimd__m128 a, b, r;
  easysimd__m128i c;

  a = easysimd_mm_loadu_ps(test_vec[0].a);
  b = easysimd_mm_loadu_ps(test_vec[0].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[0].c);
  r = easysimd_mm_fixupimm_ss(a, b, c, INT32_C(         221));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[1].a);
  b = easysimd_mm_loadu_ps(test_vec[1].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[1].c);
  r = easysimd_mm_fixupimm_ss(a, b, c, INT32_C(         198));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[2].a);
  b = easysimd_mm_loadu_ps(test_vec[2].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[2].c);
  r = easysimd_mm_fixupimm_ss(a, b, c, INT32_C(          36));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[3].a);
  b = easysimd_mm_loadu_ps(test_vec[3].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[3].c);
  r = easysimd_mm_fixupimm_ss(a, b, c, INT32_C(          25));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[4].a);
  b = easysimd_mm_loadu_ps(test_vec[4].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[4].c);
  r = easysimd_mm_fixupimm_ss(a, b, c, INT32_C(          34));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[5].a);
  b = easysimd_mm_loadu_ps(test_vec[5].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[5].c);
  r = easysimd_mm_fixupimm_ss(a, b, c, INT32_C(          98));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[6].a);
  b = easysimd_mm_loadu_ps(test_vec[6].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[6].c);
  r = easysimd_mm_fixupimm_ss(a, b, c, INT32_C(          20));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[7].a);
  b = easysimd_mm_loadu_ps(test_vec[7].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[7].c);
  r = easysimd_mm_fixupimm_ss(a, b, c, INT32_C(         252));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float32 values[8 * 2 * sizeof(easysimd__m128)];
  easysimd_test_x86_random_f32x4_full(8, 2, values, -1000.0f, 1000.0f, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m128 a = easysimd_test_x86_random_extract_f32x4(i, 2, 0, values);
    easysimd__m128 b = easysimd_test_x86_random_extract_f32x4(i, 2, 1, values);
    easysimd__m128i c = easysimd_test_x86_random_i32x4();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m128 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm_fixupimm_ss, r, easysimd_mm_setzero_ps(), imm8, a, b, c);

    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_fixupimm_ss (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const easysimd__mmask8 k;
    const easysimd_float32 b[4];
    const int32_t c[4];
    const int imm8;
    const easysimd_float32 r[4];
  } test_vec[] = {
    { {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(    17.73),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   583.28) },
      UINT8_C(236),
      { EASYSIMD_FLOAT32_C(    68.85),            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   338.17) },
      {  INT32_C(   205575547), -INT32_C(  1485220724),  INT32_C(   241697071), -INT32_C(  1133261982) },
       INT32_C(         123),
      {            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   338.17) } },
    { { EASYSIMD_FLOAT32_C(   817.25), EASYSIMD_FLOAT32_C(   -66.40), EASYSIMD_FLOAT32_C(   105.72), EASYSIMD_FLOAT32_C(   631.13) },
      UINT8_C( 17),
      { EASYSIMD_FLOAT32_C(   185.74), EASYSIMD_FLOAT32_C(   511.50), EASYSIMD_FLOAT32_C(  -859.94), EASYSIMD_FLOAT32_C(   913.76) },
      {  INT32_C(   349175174), -INT32_C(  1608832262),  INT32_C(  1981548449), -INT32_C(  1677514681) },
       INT32_C(         162),
      { EASYSIMD_FLOAT32_C(   185.74), EASYSIMD_FLOAT32_C(   511.50), EASYSIMD_FLOAT32_C(  -859.94), EASYSIMD_FLOAT32_C(   913.76) } },
    { { EASYSIMD_FLOAT32_C(  -266.07), EASYSIMD_FLOAT32_C(  -678.72), EASYSIMD_FLOAT32_C(   797.47), EASYSIMD_FLOAT32_C(  -873.06) },
      UINT8_C( 18),
      { EASYSIMD_FLOAT32_C(  -943.82), EASYSIMD_FLOAT32_C(   178.63), EASYSIMD_FLOAT32_C(  -914.24), EASYSIMD_FLOAT32_C(   532.25) },
      {  INT32_C(  1289553625), -INT32_C(   711632446), -INT32_C(   363092243), -INT32_C(  1595576203) },
       INT32_C(         139),
      { EASYSIMD_FLOAT32_C(  -266.07), EASYSIMD_FLOAT32_C(   178.63), EASYSIMD_FLOAT32_C(  -914.24), EASYSIMD_FLOAT32_C(   532.25) } },
    { { EASYSIMD_FLOAT32_C(  -158.88), EASYSIMD_FLOAT32_C(  -140.71), EASYSIMD_FLOAT32_C(   552.23), EASYSIMD_FLOAT32_C(   322.89) },
      UINT8_C(161),
      { EASYSIMD_FLOAT32_C(   792.00), EASYSIMD_FLOAT32_C(  -875.53), EASYSIMD_FLOAT32_C(   222.85), EASYSIMD_FLOAT32_C(  -731.62) },
      { -INT32_C(  1161110857),  INT32_C(  1097100406),  INT32_C(   354055951), -INT32_C(  1378326700) },
       INT32_C(         252),
      { EASYSIMD_FLOAT32_C(     0.50), EASYSIMD_FLOAT32_C(  -875.53), EASYSIMD_FLOAT32_C(   222.85), EASYSIMD_FLOAT32_C(  -731.62) } },
    { { EASYSIMD_FLOAT32_C(   142.21), EASYSIMD_FLOAT32_C(   394.02), EASYSIMD_FLOAT32_C(   851.66), EASYSIMD_FLOAT32_C(  -788.94) },
      UINT8_C(  8),
      { EASYSIMD_FLOAT32_C(   656.53), EASYSIMD_FLOAT32_C(   647.14), EASYSIMD_FLOAT32_C(   549.23), EASYSIMD_FLOAT32_C(   473.77) },
      {  INT32_C(  1786255237), -INT32_C(   118553673),  INT32_C(  1890619798), -INT32_C(   941200805) },
       INT32_C(         207),
      { EASYSIMD_FLOAT32_C(   142.21), EASYSIMD_FLOAT32_C(   647.14), EASYSIMD_FLOAT32_C(   549.23), EASYSIMD_FLOAT32_C(   473.77) } },
    { { EASYSIMD_FLOAT32_C(  -419.26), EASYSIMD_FLOAT32_C(  -345.05), EASYSIMD_FLOAT32_C(   104.91), EASYSIMD_FLOAT32_C(   766.48) },
      UINT8_C( 65),
      { EASYSIMD_FLOAT32_C(  -833.55), EASYSIMD_FLOAT32_C(   244.97), EASYSIMD_FLOAT32_C(   680.24), EASYSIMD_FLOAT32_C(   -99.63) },
      { -INT32_C(   995583252), -INT32_C(   495868856),  INT32_C(  1583839558), -INT32_C(   183119374) },
       INT32_C(         193),
      {     -EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   244.97), EASYSIMD_FLOAT32_C(   680.24), EASYSIMD_FLOAT32_C(   -99.63) } },
    { { EASYSIMD_FLOAT32_C(   566.25), EASYSIMD_FLOAT32_C(   477.71), EASYSIMD_FLOAT32_C(    27.31), EASYSIMD_FLOAT32_C(   622.43) },
      UINT8_C(190),
      { EASYSIMD_FLOAT32_C(  -343.66), EASYSIMD_FLOAT32_C(   113.07), EASYSIMD_FLOAT32_C(   154.68), EASYSIMD_FLOAT32_C(   497.46) },
      { -INT32_C(   517427717),  INT32_C(  1242101620), -INT32_C(   667530691), -INT32_C(  1759446286) },
       INT32_C(         107),
      { EASYSIMD_FLOAT32_C(   566.25), EASYSIMD_FLOAT32_C(   113.07), EASYSIMD_FLOAT32_C(   154.68), EASYSIMD_FLOAT32_C(   497.46) } },
    { { EASYSIMD_FLOAT32_C(   972.36), EASYSIMD_FLOAT32_C(  -293.10), EASYSIMD_FLOAT32_C(  -179.65), EASYSIMD_FLOAT32_C(   764.37) },
      UINT8_C(106),
      { EASYSIMD_FLOAT32_C(  -168.62), EASYSIMD_FLOAT32_C(  -956.80), EASYSIMD_FLOAT32_C(  -967.26), EASYSIMD_FLOAT32_C(   973.59) },
      {  INT32_C(  1362876219),  INT32_C(  1482685644), -INT32_C(    78439090),  INT32_C(  1030698309) },
       INT32_C(          61),
      { EASYSIMD_FLOAT32_C(   972.36), EASYSIMD_FLOAT32_C(  -956.80), EASYSIMD_FLOAT32_C(  -967.26), EASYSIMD_FLOAT32_C(   973.59) } },
  };

  easysimd__m128 a, b, r;
  easysimd__m128i c;

  a = easysimd_mm_loadu_ps(test_vec[0].a);
  b = easysimd_mm_loadu_ps(test_vec[0].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[0].c);
  r = easysimd_mm_mask_fixupimm_ss(a, test_vec[0].k, b, c, INT32_C(         123));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[1].a);
  b = easysimd_mm_loadu_ps(test_vec[1].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[1].c);
  r = easysimd_mm_mask_fixupimm_ss(a, test_vec[1].k, b, c, INT32_C(         162));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[2].a);
  b = easysimd_mm_loadu_ps(test_vec[2].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[2].c);
  r = easysimd_mm_mask_fixupimm_ss(a, test_vec[2].k, b, c, INT32_C(         139));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[3].a);
  b = easysimd_mm_loadu_ps(test_vec[3].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[3].c);
  r = easysimd_mm_mask_fixupimm_ss(a, test_vec[3].k, b, c, INT32_C(         252));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[4].a);
  b = easysimd_mm_loadu_ps(test_vec[4].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[4].c);
  r = easysimd_mm_mask_fixupimm_ss(a, test_vec[4].k, b, c, INT32_C(         207));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[5].a);
  b = easysimd_mm_loadu_ps(test_vec[5].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[5].c);
  r = easysimd_mm_mask_fixupimm_ss(a, test_vec[5].k, b, c, INT32_C(         193));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[6].a);
  b = easysimd_mm_loadu_ps(test_vec[6].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[6].c);
  r = easysimd_mm_mask_fixupimm_ss(a, test_vec[6].k, b, c, INT32_C(         107));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[7].a);
  b = easysimd_mm_loadu_ps(test_vec[7].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[7].c);
  r = easysimd_mm_mask_fixupimm_ss(a, test_vec[7].k, b, c, INT32_C(          61));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float32 values[8 * 2 * sizeof(easysimd__m128)];
  easysimd_test_x86_random_f32x4_full(8, 2, values, -1000.0f, 1000.0f, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m128 a = easysimd_test_x86_random_extract_f32x4(i, 2, 0, values);
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 b = easysimd_test_x86_random_extract_f32x4(i, 2, 1, values);
    easysimd__m128i c = easysimd_test_x86_random_i32x4();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m128 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm_mask_fixupimm_ss, r, easysimd_mm_setzero_ps(), imm8, a, k, b, c);

    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_fixupimm_ss (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const int32_t c[4];
    const int imm8;
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C(234),
      {            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -233.09),            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(   338.03) },
      { EASYSIMD_FLOAT32_C(   938.21),            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -268.29) },
      { -INT32_C(    33940159),  INT32_C(  1492514891), -INT32_C(  1015021536),  INT32_C(   172015337) },
       INT32_C(          98),
      { EASYSIMD_FLOAT32_C(     0.00),            EASYSIMD_MATH_NANF,            EASYSIMD_MATH_NANF, EASYSIMD_FLOAT32_C(  -268.29) } },
    { UINT8_C(100),
      { EASYSIMD_FLOAT32_C(   984.95), EASYSIMD_FLOAT32_C(   613.19), EASYSIMD_FLOAT32_C(   216.07), EASYSIMD_FLOAT32_C(   990.37) },
      { EASYSIMD_FLOAT32_C(   512.12), EASYSIMD_FLOAT32_C(   530.89), EASYSIMD_FLOAT32_C(  -760.11), EASYSIMD_FLOAT32_C(  -884.63) },
      { -INT32_C(  2020721377),  INT32_C(  2020846749),  INT32_C(   783899921),  INT32_C(  1333442135) },
       INT32_C(         172),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   530.89), EASYSIMD_FLOAT32_C(  -760.11), EASYSIMD_FLOAT32_C(  -884.63) } },
    { UINT8_C( 82),
      { EASYSIMD_FLOAT32_C(   252.22), EASYSIMD_FLOAT32_C(  -949.26), EASYSIMD_FLOAT32_C(   957.94), EASYSIMD_FLOAT32_C(  -357.53) },
      { EASYSIMD_FLOAT32_C(  -170.23), EASYSIMD_FLOAT32_C(   968.08), EASYSIMD_FLOAT32_C(   431.81), EASYSIMD_FLOAT32_C(  -235.42) },
      {  INT32_C(  1913754930),  INT32_C(  1411806111),  INT32_C(   410291163), -INT32_C(   961152231) },
       INT32_C(         110),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   968.08), EASYSIMD_FLOAT32_C(   431.81), EASYSIMD_FLOAT32_C(  -235.42) } },
    { UINT8_C(230),
      { EASYSIMD_FLOAT32_C(  -380.08), EASYSIMD_FLOAT32_C(   711.98), EASYSIMD_FLOAT32_C(   903.51), EASYSIMD_FLOAT32_C(  -241.52) },
      { EASYSIMD_FLOAT32_C(  -291.25), EASYSIMD_FLOAT32_C(   790.88), EASYSIMD_FLOAT32_C(   342.11), EASYSIMD_FLOAT32_C(   113.51) },
      {  INT32_C(  2140939013), -INT32_C(   497923982), -INT32_C(  1978358540),  INT32_C(   690587573) },
       INT32_C(         173),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   790.88), EASYSIMD_FLOAT32_C(   342.11), EASYSIMD_FLOAT32_C(   113.51) } },
    { UINT8_C(242),
      { EASYSIMD_FLOAT32_C(  -442.21), EASYSIMD_FLOAT32_C(   668.58), EASYSIMD_FLOAT32_C(  -548.45), EASYSIMD_FLOAT32_C(  -504.00) },
      { EASYSIMD_FLOAT32_C(   707.30), EASYSIMD_FLOAT32_C(  -856.65), EASYSIMD_FLOAT32_C(   227.71), EASYSIMD_FLOAT32_C(   692.25) },
      { -INT32_C(   772976100), -INT32_C(   268543208),  INT32_C(  1240785958), -INT32_C(   910396288) },
       INT32_C(         198),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -856.65), EASYSIMD_FLOAT32_C(   227.71), EASYSIMD_FLOAT32_C(   692.25) } },
    { UINT8_C(178),
      { EASYSIMD_FLOAT32_C(   756.54), EASYSIMD_FLOAT32_C(  -556.23), EASYSIMD_FLOAT32_C(   682.63), EASYSIMD_FLOAT32_C(   268.66) },
      { EASYSIMD_FLOAT32_C(   974.66), EASYSIMD_FLOAT32_C(   922.52), EASYSIMD_FLOAT32_C(   384.03), EASYSIMD_FLOAT32_C(   226.89) },
      {  INT32_C(  1899569223),  INT32_C(  1307567945),  INT32_C(  1902764319),  INT32_C(   696859342) },
       INT32_C(          56),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   922.52), EASYSIMD_FLOAT32_C(   384.03), EASYSIMD_FLOAT32_C(   226.89) } },
    { UINT8_C(109),
      { EASYSIMD_FLOAT32_C(   973.26), EASYSIMD_FLOAT32_C(   341.97), EASYSIMD_FLOAT32_C(   869.36), EASYSIMD_FLOAT32_C(  -196.97) },
      { EASYSIMD_FLOAT32_C(   310.05), EASYSIMD_FLOAT32_C(   301.17), EASYSIMD_FLOAT32_C(   567.61), EASYSIMD_FLOAT32_C(   929.98) },
      {  INT32_C(  1440849049),  INT32_C(   603170661),  INT32_C(   829072657), -INT32_C(   965026849) },
       INT32_C(         202),
      {      EASYSIMD_MATH_INFINITYF, EASYSIMD_FLOAT32_C(   301.17), EASYSIMD_FLOAT32_C(   567.61), EASYSIMD_FLOAT32_C(   929.98) } },
    { UINT8_C( 48),
      { EASYSIMD_FLOAT32_C(    13.15), EASYSIMD_FLOAT32_C(   471.12), EASYSIMD_FLOAT32_C(  -311.54), EASYSIMD_FLOAT32_C(   721.89) },
      { EASYSIMD_FLOAT32_C(   262.00), EASYSIMD_FLOAT32_C(  -969.44), EASYSIMD_FLOAT32_C(  -164.59), EASYSIMD_FLOAT32_C(   819.79) },
      { -INT32_C(   529893033), -INT32_C(   229006686),  INT32_C(  1535887038),  INT32_C(  1321263271) },
       INT32_C(         211),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -969.44), EASYSIMD_FLOAT32_C(  -164.59), EASYSIMD_FLOAT32_C(   819.79) } },
  };

  easysimd__m128 a, b, r;
  easysimd__m128i c;

  a = easysimd_mm_loadu_ps(test_vec[0].a);
  b = easysimd_mm_loadu_ps(test_vec[0].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[0].c);
  r = easysimd_mm_maskz_fixupimm_ss(test_vec[0].k, a, b, c, INT32_C(          98));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[1].a);
  b = easysimd_mm_loadu_ps(test_vec[1].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[1].c);
  r = easysimd_mm_maskz_fixupimm_ss(test_vec[1].k, a, b, c, INT32_C(         172));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[2].a);
  b = easysimd_mm_loadu_ps(test_vec[2].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[2].c);
  r = easysimd_mm_maskz_fixupimm_ss(test_vec[2].k, a, b, c, INT32_C(         110));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[3].a);
  b = easysimd_mm_loadu_ps(test_vec[3].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[3].c);
  r = easysimd_mm_maskz_fixupimm_ss(test_vec[3].k, a, b, c, INT32_C(         173));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[4].a);
  b = easysimd_mm_loadu_ps(test_vec[4].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[4].c);
  r = easysimd_mm_maskz_fixupimm_ss(test_vec[4].k, a, b, c, INT32_C(         198));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[5].a);
  b = easysimd_mm_loadu_ps(test_vec[5].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[5].c);
  r = easysimd_mm_maskz_fixupimm_ss(test_vec[5].k, a, b, c, INT32_C(          56));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[6].a);
  b = easysimd_mm_loadu_ps(test_vec[6].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[6].c);
  r = easysimd_mm_maskz_fixupimm_ss(test_vec[6].k, a, b, c, INT32_C(         202));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[7].a);
  b = easysimd_mm_loadu_ps(test_vec[7].b);
  c = easysimd_x_mm_loadu_epi32(test_vec[7].c);
  r = easysimd_mm_maskz_fixupimm_ss(test_vec[7].k, a, b, c, INT32_C(         211));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float32 values[8 * 2 * sizeof(easysimd__m128)];
  easysimd_test_x86_random_f32x4_full(8, 2, values, -1000.0f, 1000.0f, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_extract_f32x4(i, 2, 0, values);
    easysimd__m128 b = easysimd_test_x86_random_extract_f32x4(i, 2, 1, values);
    easysimd__m128i c = easysimd_test_x86_random_i32x4();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m128 r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm_maskz_fixupimm_ss, r, easysimd_mm_setzero_ps(), imm8, k, a, b, c);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_fixupimm_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const int64_t c[2];
    const int imm8;
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   350.82), EASYSIMD_FLOAT64_C(   216.18) },
      { EASYSIMD_FLOAT64_C(  -285.60), EASYSIMD_FLOAT64_C(  -902.21) },
      {  INT64_C(  620105188463929266),  INT64_C(  237548509961701354) },
       INT32_C(         128),
      { EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(    -1.00) } },
    { { EASYSIMD_FLOAT64_C(  -631.34), EASYSIMD_FLOAT64_C(    16.02) },
      { EASYSIMD_FLOAT64_C(  -921.86), EASYSIMD_FLOAT64_C(   718.39) },
      {  INT64_C( 2500124996503287191),  INT64_C( 8038052891779102241) },
       INT32_C(         173),
      {       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(    52.83), EASYSIMD_FLOAT64_C(   295.39) },
      { EASYSIMD_FLOAT64_C(   -93.52), EASYSIMD_FLOAT64_C(   362.45) },
      { -INT64_C( 8514580718024667197), -INT64_C( 2672496358120718762) },
       INT32_C(         217),
      { EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    90.00) } },
    { { EASYSIMD_FLOAT64_C(  -978.45), EASYSIMD_FLOAT64_C(   495.43) },
      { EASYSIMD_FLOAT64_C(  -574.94), EASYSIMD_FLOAT64_C(  -191.08) },
      { -INT64_C( 6160254793616804697), -INT64_C( 4996045170917686737) },
       INT32_C(         249),
      {       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -191.08) } },
    { { EASYSIMD_FLOAT64_C(  -199.21), EASYSIMD_FLOAT64_C(  -609.37) },
      { EASYSIMD_FLOAT64_C(   -30.32), EASYSIMD_FLOAT64_C(   210.34) },
      {  INT64_C( 8899606358546180766), -INT64_C( 6699576735940209548) },
       INT32_C(         148),
      { EASYSIMD_FLOAT64_C(    -1.00),        EASYSIMD_MATH_INFINITY } },
    { { EASYSIMD_FLOAT64_C(   308.91), EASYSIMD_FLOAT64_C(  -158.46) },
      { EASYSIMD_FLOAT64_C(   898.81), EASYSIMD_FLOAT64_C(  -511.70) },
      {  INT64_C( 4187325779917550281),  INT64_C( 4942829023948640072) },
       INT32_C(         136),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     1.57) } },
    { { EASYSIMD_FLOAT64_C(   108.10), EASYSIMD_FLOAT64_C(  -904.44) },
      { EASYSIMD_FLOAT64_C(   -20.25), EASYSIMD_FLOAT64_C(   658.89) },
      { -INT64_C( 7234476644020821166),  INT64_C( 1369725177824373804) },
       INT32_C(          76),
      { EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(    90.00) } },
    { { EASYSIMD_FLOAT64_C(  -903.66), EASYSIMD_FLOAT64_C(   871.72) },
      { EASYSIMD_FLOAT64_C(   803.52), EASYSIMD_FLOAT64_C(  -523.86) },
      { -INT64_C(  559696881086758863), -INT64_C( 6314596997044009999) },
       INT32_C(         119),
      { EASYSIMD_FLOAT64_C(179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00),             EASYSIMD_MATH_NAN } },
  };

  easysimd__m128d a, b, r;
  easysimd__m128i c;

  a = easysimd_mm_loadu_pd(test_vec[0].a);
  b = easysimd_mm_loadu_pd(test_vec[0].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[0].c);
  r = easysimd_mm_fixupimm_pd(a, b, c, INT32_C(         128));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[1].a);
  b = easysimd_mm_loadu_pd(test_vec[1].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[1].c);
  r = easysimd_mm_fixupimm_pd(a, b, c, INT32_C(         173));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[2].a);
  b = easysimd_mm_loadu_pd(test_vec[2].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[2].c);
  r = easysimd_mm_fixupimm_pd(a, b, c, INT32_C(         217));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[3].a);
  b = easysimd_mm_loadu_pd(test_vec[3].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[3].c);
  r = easysimd_mm_fixupimm_pd(a, b, c, INT32_C(         249));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[4].a);
  b = easysimd_mm_loadu_pd(test_vec[4].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[4].c);
  r = easysimd_mm_fixupimm_pd(a, b, c, INT32_C(         148));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[5].a);
  b = easysimd_mm_loadu_pd(test_vec[5].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[5].c);
  r = easysimd_mm_fixupimm_pd(a, b, c, INT32_C(         136));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[6].a);
  b = easysimd_mm_loadu_pd(test_vec[6].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[6].c);
  r = easysimd_mm_fixupimm_pd(a, b, c, INT32_C(          76));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[7].a);
  b = easysimd_mm_loadu_pd(test_vec[7].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[7].c);
  r = easysimd_mm_fixupimm_pd(a, b, c, INT32_C(         119));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float64 values[8 * 2 * sizeof(easysimd__m128d)];
  easysimd_test_x86_random_f64x2_full(8, 2, values, -1000.0f, 1000.0f, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128i c = easysimd_test_x86_random_i64x2();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m128d r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm_fixupimm_pd, r, easysimd_mm_setzero_pd(), imm8, a, b, c);

    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_fixupimm_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[2];
    const easysimd__mmask8 k;
    const easysimd_float64 b[2];
    const int64_t c[2];
    const int imm8;
    const easysimd_float64 r[2];
  } test_vec[] = {
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   105.99) },
      UINT8_C( 33),
      { EASYSIMD_FLOAT64_C(  -575.51),             EASYSIMD_MATH_NAN },
      {  INT64_C( 7552032075000655345),  INT64_C( 4000687079639204834) },
       INT32_C(         204),
      { EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(   105.99) } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   534.06) },
      UINT8_C(175),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   667.37) },
      {  INT64_C(  342064370308265702), -INT64_C( 2670388675896549221) },
       INT32_C(         174),
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(    90.00) } },
    { { EASYSIMD_FLOAT64_C(  -491.31), EASYSIMD_FLOAT64_C(  -160.12) },
      UINT8_C(187),
      { EASYSIMD_FLOAT64_C(    -1.81), EASYSIMD_FLOAT64_C(  -236.61) },
      { -INT64_C( 6412330527046097524),  INT64_C( 8935158407947465103) },
       INT32_C(         206),
      { EASYSIMD_FLOAT64_C(    -1.81),        EASYSIMD_MATH_INFINITY } },
    { { EASYSIMD_FLOAT64_C(  -759.30), EASYSIMD_FLOAT64_C(  -146.05) },
      UINT8_C(250),
      { EASYSIMD_FLOAT64_C(   812.27), EASYSIMD_FLOAT64_C(   -16.67) },
      { -INT64_C( 1501800736779250465), -INT64_C( 4793367214755945223) },
       INT32_C(         139),
      { EASYSIMD_FLOAT64_C(  -759.30), EASYSIMD_FLOAT64_C(179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00) } },
    { { EASYSIMD_FLOAT64_C(  -624.76), EASYSIMD_FLOAT64_C(    47.24) },
      UINT8_C(174),
      { EASYSIMD_FLOAT64_C(  -226.67), EASYSIMD_FLOAT64_C(   705.40) },
      {  INT64_C(  763233639629420433), -INT64_C(  883064605290950341) },
       INT32_C(         226),
      { EASYSIMD_FLOAT64_C(  -624.76), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(   806.77), EASYSIMD_FLOAT64_C(   579.23) },
      UINT8_C( 33),
      { EASYSIMD_FLOAT64_C(  -913.17), EASYSIMD_FLOAT64_C(  -666.15) },
      { -INT64_C( 4996966512722268149), -INT64_C( 6927699424696314255) },
       INT32_C(         117),
      {       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   579.23) } },
    { { EASYSIMD_FLOAT64_C(   803.80), EASYSIMD_FLOAT64_C(   745.05) },
      UINT8_C(207),
      { EASYSIMD_FLOAT64_C(   587.43), EASYSIMD_FLOAT64_C(   666.22) },
      {  INT64_C( 5967949438923600490), -INT64_C( 3148046243471625168) },
       INT32_C(         139),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00) } },
    { { EASYSIMD_FLOAT64_C(    32.34), EASYSIMD_FLOAT64_C(   783.09) },
      UINT8_C( 85),
      { EASYSIMD_FLOAT64_C(  -749.52), EASYSIMD_FLOAT64_C(  -310.76) },
      {  INT64_C( 8065850811406873902), -INT64_C( 5654830881389822289) },
       INT32_C(         212),
      { EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(   783.09) } },
  };

  easysimd__m128d a, b, r;
  easysimd__m128i c;

  a = easysimd_mm_loadu_pd(test_vec[0].a);
  b = easysimd_mm_loadu_pd(test_vec[0].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[0].c);
  r = easysimd_mm_mask_fixupimm_pd(a, test_vec[0].k, b, c, INT32_C(         204));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[1].a);
  b = easysimd_mm_loadu_pd(test_vec[1].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[1].c);
  r = easysimd_mm_mask_fixupimm_pd(a, test_vec[1].k, b, c, INT32_C(         174));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[2].a);
  b = easysimd_mm_loadu_pd(test_vec[2].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[2].c);
  r = easysimd_mm_mask_fixupimm_pd(a, test_vec[2].k, b, c, INT32_C(         206));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[3].a);
  b = easysimd_mm_loadu_pd(test_vec[3].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[3].c);
  r = easysimd_mm_mask_fixupimm_pd(a, test_vec[3].k, b, c, INT32_C(         139));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[4].a);
  b = easysimd_mm_loadu_pd(test_vec[4].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[4].c);
  r = easysimd_mm_mask_fixupimm_pd(a, test_vec[4].k, b, c, INT32_C(         226));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[5].a);
  b = easysimd_mm_loadu_pd(test_vec[5].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[5].c);
  r = easysimd_mm_mask_fixupimm_pd(a, test_vec[5].k, b, c, INT32_C(         117));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[6].a);
  b = easysimd_mm_loadu_pd(test_vec[6].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[6].c);
  r = easysimd_mm_mask_fixupimm_pd(a, test_vec[6].k, b, c, INT32_C(         139));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[7].a);
  b = easysimd_mm_loadu_pd(test_vec[7].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[7].c);
  r = easysimd_mm_mask_fixupimm_pd(a, test_vec[7].k, b, c, INT32_C(         212));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float64 values[8 * 2 * sizeof(easysimd__m128d)];
  easysimd_test_x86_random_f64x2_full(8, 2, values, -1000.0, 1000.0, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m128d a = easysimd_test_x86_random_extract_f64x2(i, 2, 0, values);
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d b = easysimd_test_x86_random_extract_f64x2(i, 2, 1, values);
    easysimd__m128i c = easysimd_test_x86_random_i64x2();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m128d r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm_mask_fixupimm_pd, r, easysimd_mm_setzero_pd(), imm8, a, k, b, c);

    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_fixupimm_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const int64_t c[2];
    const int imm8;
    const easysimd_float64 r[2];
  } test_vec[] = {
    { UINT8_C(  0),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -129.12) },
      { EASYSIMD_FLOAT64_C(  -519.59),             EASYSIMD_MATH_NAN },
      { -INT64_C(  281426097431523747),  INT64_C( 2633914727167280797) },
       INT32_C(          11),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(165),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   -35.73) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -227.67) },
      { -INT64_C( 1815578541580304333),  INT64_C(  845229627776683383) },
       INT32_C(         243),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 48),
      { EASYSIMD_FLOAT64_C(   271.31), EASYSIMD_FLOAT64_C(   636.11) },
      { EASYSIMD_FLOAT64_C(   -16.21), EASYSIMD_FLOAT64_C(  -397.26) },
      { -INT64_C( 2017676011902388247), -INT64_C( 5558947249253800661) },
       INT32_C(         115),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(  4),
      { EASYSIMD_FLOAT64_C(    87.31), EASYSIMD_FLOAT64_C(   927.62) },
      { EASYSIMD_FLOAT64_C(  -333.09), EASYSIMD_FLOAT64_C(   396.02) },
      { -INT64_C( 3251648227893633169), -INT64_C( 7989893941971710068) },
       INT32_C(          49),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 25),
      { EASYSIMD_FLOAT64_C(   766.55), EASYSIMD_FLOAT64_C(  -518.65) },
      { EASYSIMD_FLOAT64_C(  -262.06), EASYSIMD_FLOAT64_C(  -493.67) },
      { -INT64_C( 6970671218441600088), -INT64_C(  451425223845197965) },
       INT32_C(          22),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 73),
      { EASYSIMD_FLOAT64_C(  -483.15), EASYSIMD_FLOAT64_C(   162.19) },
      { EASYSIMD_FLOAT64_C(  -642.81), EASYSIMD_FLOAT64_C(  -817.88) },
      {  INT64_C( 4497596849686573129),  INT64_C( 8483488956932326274) },
       INT32_C(         171),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(104),
      { EASYSIMD_FLOAT64_C(  -703.70), EASYSIMD_FLOAT64_C(  -186.94) },
      { EASYSIMD_FLOAT64_C(   488.77), EASYSIMD_FLOAT64_C(   365.22) },
      { -INT64_C( 7438058644691805829),  INT64_C( 3984923209591034075) },
       INT32_C(         180),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(113),
      { EASYSIMD_FLOAT64_C(   211.88), EASYSIMD_FLOAT64_C(   137.44) },
      { EASYSIMD_FLOAT64_C(  -621.87), EASYSIMD_FLOAT64_C(  -410.74) },
      { -INT64_C( 2876808472297833986), -INT64_C( 3412731146393126024) },
       INT32_C(          82),
      { EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     0.00) } },
  };

  easysimd__m128d a, b, r;
  easysimd__m128i c;

  a = easysimd_mm_loadu_pd(test_vec[0].a);
  b = easysimd_mm_loadu_pd(test_vec[0].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[0].c);
  r = easysimd_mm_maskz_fixupimm_pd(test_vec[0].k, a, b, c, INT32_C(          11));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[1].a);
  b = easysimd_mm_loadu_pd(test_vec[1].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[1].c);
  r = easysimd_mm_maskz_fixupimm_pd(test_vec[1].k, a, b, c, INT32_C(         243));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[2].a);
  b = easysimd_mm_loadu_pd(test_vec[2].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[2].c);
  r = easysimd_mm_maskz_fixupimm_pd(test_vec[2].k, a, b, c, INT32_C(         115));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[3].a);
  b = easysimd_mm_loadu_pd(test_vec[3].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[3].c);
  r = easysimd_mm_maskz_fixupimm_pd(test_vec[3].k, a, b, c, INT32_C(          49));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[4].a);
  b = easysimd_mm_loadu_pd(test_vec[4].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[4].c);
  r = easysimd_mm_maskz_fixupimm_pd(test_vec[4].k, a, b, c, INT32_C(          22));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[5].a);
  b = easysimd_mm_loadu_pd(test_vec[5].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[5].c);
  r = easysimd_mm_maskz_fixupimm_pd(test_vec[5].k, a, b, c, INT32_C(         171));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[6].a);
  b = easysimd_mm_loadu_pd(test_vec[6].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[6].c);
  r = easysimd_mm_maskz_fixupimm_pd(test_vec[6].k, a, b, c, INT32_C(         180));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[7].a);
  b = easysimd_mm_loadu_pd(test_vec[7].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[7].c);
  r = easysimd_mm_maskz_fixupimm_pd(test_vec[7].k, a, b, c, INT32_C(          82));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float64 values[8 * 2 * sizeof(easysimd__m128d)];
  easysimd_test_x86_random_f64x2_full(8, 2, values, -1000.0, 1000.0, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_extract_f64x2(i, 2, 0, values);
    easysimd__m128d b = easysimd_test_x86_random_extract_f64x2(i, 2, 1, values);
    easysimd__m128i c = easysimd_test_x86_random_i64x2();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m128d r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm_maskz_fixupimm_pd, r, easysimd_mm_setzero_pd(), imm8, k, a, b, c);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_fixupimm_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[4];
    const easysimd_float64 b[4];
    const int64_t c[4];
    const int imm8;
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -357.05), EASYSIMD_FLOAT64_C(  -918.87), EASYSIMD_FLOAT64_C(   581.22), EASYSIMD_FLOAT64_C(  -739.49) },
      { EASYSIMD_FLOAT64_C(   180.13), EASYSIMD_FLOAT64_C(   663.22), EASYSIMD_FLOAT64_C(   915.28), EASYSIMD_FLOAT64_C(   681.12) },
      { -INT64_C( 9128649742646493234),  INT64_C( 5059336960335398722),  INT64_C( 4188736807440544546),  INT64_C( 6499756994776703759) },
       INT32_C(           5),
      { EASYSIMD_FLOAT64_C(     0.50), EASYSIMD_FLOAT64_C(179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00), EASYSIMD_FLOAT64_C(-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00), EASYSIMD_FLOAT64_C(     0.50) } },
    { { EASYSIMD_FLOAT64_C(   845.10), EASYSIMD_FLOAT64_C(  -581.85), EASYSIMD_FLOAT64_C(  -870.84), EASYSIMD_FLOAT64_C(   -81.12) },
      { EASYSIMD_FLOAT64_C(  -772.35), EASYSIMD_FLOAT64_C(   441.58), EASYSIMD_FLOAT64_C(  -437.87), EASYSIMD_FLOAT64_C(    89.42) },
      { -INT64_C( 7333634065308141539), -INT64_C( 8443164091348657389),  INT64_C( 2456988275574048417), -INT64_C( 1882724784740693410) },
       INT32_C(         239),
      { EASYSIMD_FLOAT64_C(   845.10),             EASYSIMD_MATH_NAN,       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(  -481.27), EASYSIMD_FLOAT64_C(   193.06), EASYSIMD_FLOAT64_C(   240.89), EASYSIMD_FLOAT64_C(  -483.65) },
      { EASYSIMD_FLOAT64_C(  -464.48), EASYSIMD_FLOAT64_C(    44.75), EASYSIMD_FLOAT64_C(  -825.70), EASYSIMD_FLOAT64_C(   -45.71) },
      {  INT64_C(  948894183572256228),  INT64_C( 1588051369400419104), -INT64_C( 8095746570882127589), -INT64_C( 3894994767606457472) },
       INT32_C(         103),
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     1.00) } },
    { { EASYSIMD_FLOAT64_C(   889.79), EASYSIMD_FLOAT64_C(   985.16), EASYSIMD_FLOAT64_C(  -982.05), EASYSIMD_FLOAT64_C(   396.04) },
      { EASYSIMD_FLOAT64_C(   101.30), EASYSIMD_FLOAT64_C(   359.33), EASYSIMD_FLOAT64_C(  -865.24), EASYSIMD_FLOAT64_C(  -895.86) },
      {  INT64_C( 5685826328924848835),  INT64_C( 5762264922207456829),  INT64_C( 8712888705034545631),  INT64_C( 4370720870943070943) },
       INT32_C(         238),
      { EASYSIMD_FLOAT64_C(     1.57), EASYSIMD_FLOAT64_C(     0.50), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(    90.00) } },
    { { EASYSIMD_FLOAT64_C(  -307.81), EASYSIMD_FLOAT64_C(  -351.00), EASYSIMD_FLOAT64_C(  -329.27), EASYSIMD_FLOAT64_C(  -413.49) },
      { EASYSIMD_FLOAT64_C(   330.82), EASYSIMD_FLOAT64_C(   968.87), EASYSIMD_FLOAT64_C(  -923.74), EASYSIMD_FLOAT64_C(    25.92) },
      { -INT64_C( 2388488068827261005), -INT64_C( 2935140386255485752), -INT64_C( 7350950240086398156), -INT64_C( 2469209362758163409) },
       INT32_C(         115),
      { EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     1.00),             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(    90.00) } },
    { { EASYSIMD_FLOAT64_C(   328.99), EASYSIMD_FLOAT64_C(   555.32), EASYSIMD_FLOAT64_C(   248.71), EASYSIMD_FLOAT64_C(  -369.68) },
      { EASYSIMD_FLOAT64_C(  -688.83), EASYSIMD_FLOAT64_C(   349.12), EASYSIMD_FLOAT64_C(   491.20), EASYSIMD_FLOAT64_C(   479.23) },
      { -INT64_C( 5041403720231609422),  INT64_C( 6013166599246382467), -INT64_C( 7182782685479161942),  INT64_C( 9180203896000963819) },
       INT32_C(         189),
      {       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(     0.50), EASYSIMD_FLOAT64_C(   491.20), EASYSIMD_FLOAT64_C(-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00) } },
    { { EASYSIMD_FLOAT64_C(  -868.10), EASYSIMD_FLOAT64_C(  -928.64), EASYSIMD_FLOAT64_C(   551.98), EASYSIMD_FLOAT64_C(  -554.35) },
      { EASYSIMD_FLOAT64_C(  -389.91), EASYSIMD_FLOAT64_C(  -482.11), EASYSIMD_FLOAT64_C(   225.65), EASYSIMD_FLOAT64_C(  -167.90) },
      {  INT64_C( 1056411706322823905),  INT64_C( 3145330439249132225), -INT64_C( 5217120311582257896), -INT64_C( 3778956683905394562) },
       INT32_C(          29),
      { EASYSIMD_FLOAT64_C(     0.50), EASYSIMD_FLOAT64_C(    90.00), EASYSIMD_FLOAT64_C(     1.57),             EASYSIMD_MATH_NAN } },
    { { EASYSIMD_FLOAT64_C(   385.51), EASYSIMD_FLOAT64_C(   259.07), EASYSIMD_FLOAT64_C(  -716.83), EASYSIMD_FLOAT64_C(  -360.30) },
      { EASYSIMD_FLOAT64_C(   249.22), EASYSIMD_FLOAT64_C(  -180.84), EASYSIMD_FLOAT64_C(   320.04), EASYSIMD_FLOAT64_C(  -457.43) },
      {  INT64_C( 8507632569968538148), -INT64_C( 5931405894810227906),  INT64_C( 6480625695532269019), -INT64_C( 4170117051513769010) },
       INT32_C(         118),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(     1.57), EASYSIMD_FLOAT64_C(-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00), EASYSIMD_FLOAT64_C(     0.00) } },
  };

  easysimd__m256d a, b, r;
  easysimd__m256i c;

  a = easysimd_mm256_loadu_pd(test_vec[0].a);
  b = easysimd_mm256_loadu_pd(test_vec[0].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[0].c);
  r = easysimd_mm256_fixupimm_pd(a, b, c, INT32_C(           5));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[1].a);
  b = easysimd_mm256_loadu_pd(test_vec[1].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[1].c);
  r = easysimd_mm256_fixupimm_pd(a, b, c, INT32_C(         239));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[2].a);
  b = easysimd_mm256_loadu_pd(test_vec[2].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[2].c);
  r = easysimd_mm256_fixupimm_pd(a, b, c, INT32_C(         103));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[3].a);
  b = easysimd_mm256_loadu_pd(test_vec[3].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[3].c);
  r = easysimd_mm256_fixupimm_pd(a, b, c, INT32_C(         238));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[4].a);
  b = easysimd_mm256_loadu_pd(test_vec[4].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[4].c);
  r = easysimd_mm256_fixupimm_pd(a, b, c, INT32_C(         115));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[5].a);
  b = easysimd_mm256_loadu_pd(test_vec[5].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[5].c);
  r = easysimd_mm256_fixupimm_pd(a, b, c, INT32_C(         189));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[6].a);
  b = easysimd_mm256_loadu_pd(test_vec[6].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[6].c);
  r = easysimd_mm256_fixupimm_pd(a, b, c, INT32_C(          29));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[7].a);
  b = easysimd_mm256_loadu_pd(test_vec[7].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[7].c);
  r = easysimd_mm256_fixupimm_pd(a, b, c, INT32_C(         118));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float64 values[8 * 2 * sizeof(easysimd__m256d)];
  easysimd_test_x86_random_f64x4_full(8, 2, values, -1000.0, 1000.0, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256i c = easysimd_test_x86_random_i64x4();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m256d r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm256_fixupimm_pd, r, easysimd_mm256_setzero_pd(), imm8, a, b, c);

    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_fixupimm_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[4];
    const easysimd__mmask8 k;
    const easysimd_float64 b[4];
    const int64_t c[4];
    const int imm8;
    const easysimd_float64 r[4];
  } test_vec[] = {
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   284.54),             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -557.46) },
      UINT8_C( 51),
      { EASYSIMD_FLOAT64_C(   147.63),             EASYSIMD_MATH_NAN,             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -256.43) },
      { -INT64_C( 8492544636006501984),  INT64_C( 1380093874916417348),  INT64_C( 9015491327522902615), -INT64_C(  877214211027671538) },
       INT32_C(         115),
      {             EASYSIMD_MATH_NAN,       -EASYSIMD_MATH_INFINITY,             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -557.46) } },
    { { EASYSIMD_FLOAT64_C(  -847.12), EASYSIMD_FLOAT64_C(   629.14), EASYSIMD_FLOAT64_C(  -409.74), EASYSIMD_FLOAT64_C(   115.91) },
      UINT8_C(236),
      { EASYSIMD_FLOAT64_C(  -249.78), EASYSIMD_FLOAT64_C(   -62.78), EASYSIMD_FLOAT64_C(   -45.54), EASYSIMD_FLOAT64_C(   873.46) },
      { -INT64_C( 3662723125764578246),  INT64_C( 5924144313530597259), -INT64_C(  387590526236314650), -INT64_C( 2071225192278027565) },
       INT32_C(          43),
      { EASYSIMD_FLOAT64_C(  -847.12), EASYSIMD_FLOAT64_C(   629.14), EASYSIMD_FLOAT64_C(     1.57),        EASYSIMD_MATH_INFINITY } },
    { { EASYSIMD_FLOAT64_C(  -392.43), EASYSIMD_FLOAT64_C(  -128.70), EASYSIMD_FLOAT64_C(  -617.79), EASYSIMD_FLOAT64_C(   536.99) },
      UINT8_C(214),
      { EASYSIMD_FLOAT64_C(    71.31), EASYSIMD_FLOAT64_C(   530.66), EASYSIMD_FLOAT64_C(   715.63), EASYSIMD_FLOAT64_C(  -909.53) },
      { -INT64_C( 6970163927411674839), -INT64_C(  620701836407105738),  INT64_C( 5885395240567331954), -INT64_C( 5212810402148347552) },
       INT32_C(          17),
      { EASYSIMD_FLOAT64_C(  -392.43), EASYSIMD_FLOAT64_C(179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00), EASYSIMD_FLOAT64_C(  -617.79), EASYSIMD_FLOAT64_C(   536.99) } },
    { { EASYSIMD_FLOAT64_C(   999.16), EASYSIMD_FLOAT64_C(  -563.16), EASYSIMD_FLOAT64_C(   426.20), EASYSIMD_FLOAT64_C(  -794.64) },
      UINT8_C( 10),
      { EASYSIMD_FLOAT64_C(   646.28), EASYSIMD_FLOAT64_C(    28.53), EASYSIMD_FLOAT64_C(   783.47), EASYSIMD_FLOAT64_C(  -418.89) },
      { -INT64_C( 2381513542705576348), -INT64_C( 3120337352039273844), -INT64_C( 2368323309043186942),  INT64_C( 6756482433063967150) },
       INT32_C(          95),
      { EASYSIMD_FLOAT64_C(   999.16), EASYSIMD_FLOAT64_C(179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00), EASYSIMD_FLOAT64_C(   426.20), EASYSIMD_FLOAT64_C(    -1.00) } },
    { { EASYSIMD_FLOAT64_C(  -686.93), EASYSIMD_FLOAT64_C(  -996.89), EASYSIMD_FLOAT64_C(    23.64), EASYSIMD_FLOAT64_C(   460.69) },
      UINT8_C( 46),
      { EASYSIMD_FLOAT64_C(  -734.22), EASYSIMD_FLOAT64_C(  -172.41), EASYSIMD_FLOAT64_C(  -795.74), EASYSIMD_FLOAT64_C(  -581.33) },
      { -INT64_C( 1158653695890929332), -INT64_C(  443457283080208555), -INT64_C( 9117148594293917785),  INT64_C( 8959431178916517668) },
       INT32_C(         148),
      { EASYSIMD_FLOAT64_C(  -686.93), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    23.64), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(  -543.28), EASYSIMD_FLOAT64_C(  -205.48), EASYSIMD_FLOAT64_C(   534.58), EASYSIMD_FLOAT64_C(   206.94) },
      UINT8_C(179),
      { EASYSIMD_FLOAT64_C(   731.75), EASYSIMD_FLOAT64_C(  -510.96), EASYSIMD_FLOAT64_C(    80.40), EASYSIMD_FLOAT64_C(  -660.68) },
      { -INT64_C( 7759315063881065058),  INT64_C( 6023771153694935608),  INT64_C( 4916981884698237237),  INT64_C(  628823249951242487) },
       INT32_C(          37),
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -205.48), EASYSIMD_FLOAT64_C(   534.58), EASYSIMD_FLOAT64_C(   206.94) } },
    { { EASYSIMD_FLOAT64_C(   360.33), EASYSIMD_FLOAT64_C(   462.61), EASYSIMD_FLOAT64_C(   876.31), EASYSIMD_FLOAT64_C(  -568.36) },
      UINT8_C( 35),
      { EASYSIMD_FLOAT64_C(    -6.73), EASYSIMD_FLOAT64_C(   591.94), EASYSIMD_FLOAT64_C(  -477.89), EASYSIMD_FLOAT64_C(    -7.57) },
      { -INT64_C( 4687038835906066249),  INT64_C( 6987334546953029604),  INT64_C( 2044174724947721019), -INT64_C( 8935575087944210919) },
       INT32_C(          61),
      { EASYSIMD_FLOAT64_C(    -6.73), EASYSIMD_FLOAT64_C(-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00), EASYSIMD_FLOAT64_C(   876.31), EASYSIMD_FLOAT64_C(  -568.36) } },
    { { EASYSIMD_FLOAT64_C(  -971.22), EASYSIMD_FLOAT64_C(   948.30), EASYSIMD_FLOAT64_C(   197.80), EASYSIMD_FLOAT64_C(   675.06) },
      UINT8_C(165),
      { EASYSIMD_FLOAT64_C(   -23.17), EASYSIMD_FLOAT64_C(   -18.73), EASYSIMD_FLOAT64_C(  -743.83), EASYSIMD_FLOAT64_C(   289.90) },
      { -INT64_C( 1044685636867850631), -INT64_C( 7652143293153812354), -INT64_C( 6867556035621623170),  INT64_C( 3713250581274244488) },
       INT32_C(          27),
      { EASYSIMD_FLOAT64_C(  -971.22), EASYSIMD_FLOAT64_C(   948.30),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   675.06) } },
  };

  easysimd__m256d a, b, r;
  easysimd__m256i c;

  a = easysimd_mm256_loadu_pd(test_vec[0].a);
  b = easysimd_mm256_loadu_pd(test_vec[0].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[0].c);
  r = easysimd_mm256_mask_fixupimm_pd(a, test_vec[0].k, b, c, INT32_C(         115));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[1].a);
  b = easysimd_mm256_loadu_pd(test_vec[1].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[1].c);
  r = easysimd_mm256_mask_fixupimm_pd(a, test_vec[1].k, b, c, INT32_C(          43));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[2].a);
  b = easysimd_mm256_loadu_pd(test_vec[2].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[2].c);
  r = easysimd_mm256_mask_fixupimm_pd(a, test_vec[2].k, b, c, INT32_C(          17));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[3].a);
  b = easysimd_mm256_loadu_pd(test_vec[3].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[3].c);
  r = easysimd_mm256_mask_fixupimm_pd(a, test_vec[3].k, b, c, INT32_C(          95));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[4].a);
  b = easysimd_mm256_loadu_pd(test_vec[4].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[4].c);
  r = easysimd_mm256_mask_fixupimm_pd(a, test_vec[4].k, b, c, INT32_C(         148));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[5].a);
  b = easysimd_mm256_loadu_pd(test_vec[5].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[5].c);
  r = easysimd_mm256_mask_fixupimm_pd(a, test_vec[5].k, b, c, INT32_C(          37));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[6].a);
  b = easysimd_mm256_loadu_pd(test_vec[6].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[6].c);
  r = easysimd_mm256_mask_fixupimm_pd(a, test_vec[6].k, b, c, INT32_C(          61));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[7].a);
  b = easysimd_mm256_loadu_pd(test_vec[7].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[7].c);
  r = easysimd_mm256_mask_fixupimm_pd(a, test_vec[7].k, b, c, INT32_C(          27));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float64 values[8 * 2 * sizeof(easysimd__m256d)];
  easysimd_test_x86_random_f64x4_full(8, 2, values, -1000.0, 1000.0, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m256d a = easysimd_test_x86_random_extract_f64x4(i, 2, 0, values);
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d b = easysimd_test_x86_random_extract_f64x4(i, 2, 1, values);
    easysimd__m256i c = easysimd_test_x86_random_i64x4();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m256d r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm256_mask_fixupimm_pd, r, easysimd_mm256_setzero_pd(), imm8, a, k, b, c);

    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_fixupimm_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[4];
    const easysimd_float64 b[4];
    const int64_t c[4];
    const int imm8;
    const easysimd_float64 r[4];
  } test_vec[] = {
    { UINT8_C(216),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -626.70),             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   918.31) },
      { EASYSIMD_FLOAT64_C(   902.42),             EASYSIMD_MATH_NAN,             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(    27.88) },
      { -INT64_C( 8812304650163596194),  INT64_C( 8309787187671250463), -INT64_C( 7426664054712978884), -INT64_C(  262939828427731939) },
       INT32_C(         173),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 58),
      { EASYSIMD_FLOAT64_C(  -815.50), EASYSIMD_FLOAT64_C(  -361.54), EASYSIMD_FLOAT64_C(   848.47), EASYSIMD_FLOAT64_C(   731.77) },
      { EASYSIMD_FLOAT64_C(   679.70), EASYSIMD_FLOAT64_C(  -364.13), EASYSIMD_FLOAT64_C(     8.88), EASYSIMD_FLOAT64_C(  -864.82) },
      {  INT64_C( 6200296752056750058),  INT64_C( 6433816659149094451), -INT64_C(  117947748737162894),  INT64_C( 8673641022723217489) },
       INT32_C(         197),
      { EASYSIMD_FLOAT64_C(     0.00),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(     0.00),       -EASYSIMD_MATH_INFINITY } },
    { UINT8_C(153),
      { EASYSIMD_FLOAT64_C(   416.76), EASYSIMD_FLOAT64_C(  -669.59), EASYSIMD_FLOAT64_C(  -702.63), EASYSIMD_FLOAT64_C(  -551.63) },
      { EASYSIMD_FLOAT64_C(  -612.49), EASYSIMD_FLOAT64_C(  -123.67), EASYSIMD_FLOAT64_C(   348.16), EASYSIMD_FLOAT64_C(  -520.38) },
      {  INT64_C( 4308452503886505507), -INT64_C( 7997711679663299588), -INT64_C( 6652653733922340409),  INT64_C( 6968256358552970596) },
       INT32_C(          94),
      { EASYSIMD_FLOAT64_C(  -612.49), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.50) } },
    { UINT8_C( 51),
      { EASYSIMD_FLOAT64_C(   164.04), EASYSIMD_FLOAT64_C(   301.20), EASYSIMD_FLOAT64_C(   598.30), EASYSIMD_FLOAT64_C(  -566.94) },
      { EASYSIMD_FLOAT64_C(  -590.81), EASYSIMD_FLOAT64_C(   168.61), EASYSIMD_FLOAT64_C(   -97.20), EASYSIMD_FLOAT64_C(  -439.59) },
      {  INT64_C( 7820515050576106443),  INT64_C(  125348702409922831),  INT64_C(   28007619820347392), -INT64_C( 7932678677096053707) },
       INT32_C(         201),
      { EASYSIMD_FLOAT64_C(179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00), EASYSIMD_FLOAT64_C(   168.61), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 38),
      { EASYSIMD_FLOAT64_C(   541.92), EASYSIMD_FLOAT64_C(  -132.48), EASYSIMD_FLOAT64_C(  -521.28), EASYSIMD_FLOAT64_C(   444.34) },
      { EASYSIMD_FLOAT64_C(  -415.11), EASYSIMD_FLOAT64_C(   168.63), EASYSIMD_FLOAT64_C(  -527.78), EASYSIMD_FLOAT64_C(  -230.61) },
      { -INT64_C( 4892239152882043385),  INT64_C(  156750231931526147),  INT64_C(  202923168954264902), -INT64_C( 1723305389559356970) },
       INT32_C(         204),
      { EASYSIMD_FLOAT64_C(     0.00),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(145),
      { EASYSIMD_FLOAT64_C(   807.09), EASYSIMD_FLOAT64_C(  -679.31), EASYSIMD_FLOAT64_C(  -498.84), EASYSIMD_FLOAT64_C(   486.79) },
      { EASYSIMD_FLOAT64_C(   -43.44), EASYSIMD_FLOAT64_C(   510.04), EASYSIMD_FLOAT64_C(   621.97), EASYSIMD_FLOAT64_C(  -626.67) },
      { -INT64_C( 7871455767103737308), -INT64_C( 1666644143514270683),  INT64_C( 4824673154926305459),  INT64_C(  693348618212400323) },
       INT32_C(          54),
      { EASYSIMD_FLOAT64_C(   807.09), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 44),
      { EASYSIMD_FLOAT64_C(   840.44), EASYSIMD_FLOAT64_C(   919.34), EASYSIMD_FLOAT64_C(  -178.30), EASYSIMD_FLOAT64_C(  -772.04) },
      { EASYSIMD_FLOAT64_C(  -204.33), EASYSIMD_FLOAT64_C(  -830.14), EASYSIMD_FLOAT64_C(  -292.43), EASYSIMD_FLOAT64_C(   959.71) },
      {  INT64_C( 3659917210367915787),  INT64_C( 2238000330235556481), -INT64_C( 6651832730874916370),  INT64_C( 1004793110982146192) },
       INT32_C(          53),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    90.00) } },
    { UINT8_C(230),
      { EASYSIMD_FLOAT64_C(   471.06), EASYSIMD_FLOAT64_C(  -694.13), EASYSIMD_FLOAT64_C(  -607.23), EASYSIMD_FLOAT64_C(   880.25) },
      { EASYSIMD_FLOAT64_C(   474.49), EASYSIMD_FLOAT64_C(   295.56), EASYSIMD_FLOAT64_C(  -559.34), EASYSIMD_FLOAT64_C(    16.40) },
      { -INT64_C( 7580743870084302937), -INT64_C( 4789490465420023080), -INT64_C( 4271362955824714498),  INT64_C( 5416843691555407363) },
       INT32_C(          54),
      { EASYSIMD_FLOAT64_C(     0.00),             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     0.00) } },
  };

  easysimd__m256d a, b, r;
  easysimd__m256i c;

  a = easysimd_mm256_loadu_pd(test_vec[0].a);
  b = easysimd_mm256_loadu_pd(test_vec[0].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[0].c);
  r = easysimd_mm256_maskz_fixupimm_pd(test_vec[0].k, a, b, c, INT32_C(         173));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[1].a);
  b = easysimd_mm256_loadu_pd(test_vec[1].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[1].c);
  r = easysimd_mm256_maskz_fixupimm_pd(test_vec[1].k, a, b, c, INT32_C(         197));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[2].a);
  b = easysimd_mm256_loadu_pd(test_vec[2].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[2].c);
  r = easysimd_mm256_maskz_fixupimm_pd(test_vec[2].k, a, b, c, INT32_C(          94));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[3].a);
  b = easysimd_mm256_loadu_pd(test_vec[3].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[3].c);
  r = easysimd_mm256_maskz_fixupimm_pd(test_vec[3].k, a, b, c, INT32_C(         201));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[4].a);
  b = easysimd_mm256_loadu_pd(test_vec[4].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[4].c);
  r = easysimd_mm256_maskz_fixupimm_pd(test_vec[4].k, a, b, c, INT32_C(         204));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[5].a);
  b = easysimd_mm256_loadu_pd(test_vec[5].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[5].c);
  r = easysimd_mm256_maskz_fixupimm_pd(test_vec[5].k, a, b, c, INT32_C(          54));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[6].a);
  b = easysimd_mm256_loadu_pd(test_vec[6].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[6].c);
  r = easysimd_mm256_maskz_fixupimm_pd(test_vec[6].k, a, b, c, INT32_C(          53));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[7].a);
  b = easysimd_mm256_loadu_pd(test_vec[7].b);
  c = easysimd_x_mm256_loadu_epi64(test_vec[7].c);
  r = easysimd_mm256_maskz_fixupimm_pd(test_vec[7].k, a, b, c, INT32_C(          54));
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float64 values[8 * 2 * sizeof(easysimd__m256d)];
  easysimd_test_x86_random_f64x4_full(8, 2, values, -1000.0, 1000.0, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_extract_f64x4(i, 2, 0, values);
    easysimd__m256d b = easysimd_test_x86_random_extract_f64x4(i, 2, 1, values);
    easysimd__m256i c = easysimd_test_x86_random_i64x4();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m256d r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm256_maskz_fixupimm_pd, r, easysimd_mm256_setzero_pd(), imm8, k, a, b, c);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_fixupimm_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const int64_t c[8];
    const int imm8;
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   390.06), EASYSIMD_FLOAT64_C(   302.89), EASYSIMD_FLOAT64_C(  -234.09), EASYSIMD_FLOAT64_C(   130.83),
        EASYSIMD_FLOAT64_C(  -945.44), EASYSIMD_FLOAT64_C(   529.48), EASYSIMD_FLOAT64_C(   338.06), EASYSIMD_FLOAT64_C(   661.45) },
      { EASYSIMD_FLOAT64_C(   521.57), EASYSIMD_FLOAT64_C(    95.82), EASYSIMD_FLOAT64_C(  -841.68), EASYSIMD_FLOAT64_C(  -470.21),
        EASYSIMD_FLOAT64_C(   561.99), EASYSIMD_FLOAT64_C(   530.33), EASYSIMD_FLOAT64_C(   956.70), EASYSIMD_FLOAT64_C(   586.00) },
      { -INT64_C( 6251646270719399674), -INT64_C( 9043880886728127943), -INT64_C( 2598222736183392795), -INT64_C( 1860063746138400589),
         INT64_C( 8656489402950389410),  INT64_C( 6184841349590401554),  INT64_C( 4326059874980939774),  INT64_C( 4224141810940501977) },
       INT32_C(         185),
      { EASYSIMD_FLOAT64_C(179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00), EASYSIMD_FLOAT64_C(   302.89), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00),
        EASYSIMD_FLOAT64_C(    -0.00),        EASYSIMD_MATH_INFINITY,             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(     0.50) } },
    { { EASYSIMD_FLOAT64_C(   294.30), EASYSIMD_FLOAT64_C(   -26.26), EASYSIMD_FLOAT64_C(  -283.92), EASYSIMD_FLOAT64_C(  -336.89),
        EASYSIMD_FLOAT64_C(   689.15), EASYSIMD_FLOAT64_C(   908.59), EASYSIMD_FLOAT64_C(   332.56), EASYSIMD_FLOAT64_C(   893.04) },
      { EASYSIMD_FLOAT64_C(   358.89), EASYSIMD_FLOAT64_C(  -307.26), EASYSIMD_FLOAT64_C(  -111.71), EASYSIMD_FLOAT64_C(   832.92),
        EASYSIMD_FLOAT64_C(   874.46), EASYSIMD_FLOAT64_C(  -270.54), EASYSIMD_FLOAT64_C(  -139.87), EASYSIMD_FLOAT64_C(  -784.21) },
      { -INT64_C( 7962448834612225555), -INT64_C(  392236992424108323), -INT64_C( 3462612895072911789), -INT64_C( 1699332621576613890),
        -INT64_C( 5736409637550415398),  INT64_C(  130275286591090508), -INT64_C( 7094050535347827501), -INT64_C( 2733130823568114131) },
       INT32_C(          24),
      { EASYSIMD_FLOAT64_C(    90.00),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00),
        EASYSIMD_FLOAT64_C(     1.57),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(     1.57), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(  -319.26), EASYSIMD_FLOAT64_C(   718.51), EASYSIMD_FLOAT64_C(  -308.55), EASYSIMD_FLOAT64_C(  -594.46),
        EASYSIMD_FLOAT64_C(  -122.80), EASYSIMD_FLOAT64_C(  -227.21), EASYSIMD_FLOAT64_C(   366.12), EASYSIMD_FLOAT64_C(   421.38) },
      { EASYSIMD_FLOAT64_C(   146.34), EASYSIMD_FLOAT64_C(  -143.23), EASYSIMD_FLOAT64_C(  -675.20), EASYSIMD_FLOAT64_C(  -471.82),
        EASYSIMD_FLOAT64_C(   540.59), EASYSIMD_FLOAT64_C(   276.61), EASYSIMD_FLOAT64_C(  -173.11), EASYSIMD_FLOAT64_C(  -249.85) },
      {  INT64_C( 1229467386300978362), -INT64_C( 3608341113558327063),  INT64_C( 2510778146467776681),  INT64_C( 6836450645469087983),
         INT64_C( 2300025728764560583),  INT64_C( 1738966078427551349),  INT64_C( 3108287743583725717), -INT64_C( 8528090630843019613) },
       INT32_C(         102),
      { EASYSIMD_FLOAT64_C(179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00), EASYSIMD_FLOAT64_C(  -143.23), EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    -0.00),
        EASYSIMD_FLOAT64_C(     1.57),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   366.12), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(  -578.97), EASYSIMD_FLOAT64_C(  -883.16), EASYSIMD_FLOAT64_C(  -877.12), EASYSIMD_FLOAT64_C(   873.37),
        EASYSIMD_FLOAT64_C(  -596.71), EASYSIMD_FLOAT64_C(   351.98), EASYSIMD_FLOAT64_C(    68.79), EASYSIMD_FLOAT64_C(  -808.35) },
      { EASYSIMD_FLOAT64_C(   360.01), EASYSIMD_FLOAT64_C(  -311.99), EASYSIMD_FLOAT64_C(   200.43), EASYSIMD_FLOAT64_C(  -161.58),
        EASYSIMD_FLOAT64_C(   136.60), EASYSIMD_FLOAT64_C(  -693.17), EASYSIMD_FLOAT64_C(  -440.00), EASYSIMD_FLOAT64_C(   780.66) },
      {  INT64_C( 3347301987161906825),  INT64_C( 5198588865035049763), -INT64_C( 6692900195602079641),  INT64_C(   10023661517916473),
         INT64_C(  492477733401387322),  INT64_C(  867987076754214076),  INT64_C( 5617313734110677654), -INT64_C( 1515741989530954625) },
       INT32_C(         226),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   873.37),
        EASYSIMD_FLOAT64_C(     1.00),             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -440.00), EASYSIMD_FLOAT64_C(-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00) } },
    { { EASYSIMD_FLOAT64_C(  -995.65), EASYSIMD_FLOAT64_C(   347.22), EASYSIMD_FLOAT64_C(   849.40), EASYSIMD_FLOAT64_C(   961.46),
        EASYSIMD_FLOAT64_C(   742.43), EASYSIMD_FLOAT64_C(   -55.35), EASYSIMD_FLOAT64_C(   852.52), EASYSIMD_FLOAT64_C(  -319.99) },
      { EASYSIMD_FLOAT64_C(   -34.92), EASYSIMD_FLOAT64_C(  -901.43), EASYSIMD_FLOAT64_C(  -723.20), EASYSIMD_FLOAT64_C(  -966.42),
        EASYSIMD_FLOAT64_C(  -427.04), EASYSIMD_FLOAT64_C(   -63.90), EASYSIMD_FLOAT64_C(   502.12), EASYSIMD_FLOAT64_C(  -655.52) },
      { -INT64_C( 5663388675417969167),  INT64_C( 5134716563632119451),  INT64_C( 8799487667835833747), -INT64_C( 1141070973733060000),
         INT64_C( 2141680984098137862),  INT64_C( 1508666849613206565),  INT64_C( 2867870228952037609),  INT64_C(  966317920808529275) },
       INT32_C(         244),
      { EASYSIMD_FLOAT64_C(  -995.65), EASYSIMD_FLOAT64_C(     1.57),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(     0.50),
        EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(    -0.00),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(    -1.00) } },
    { { EASYSIMD_FLOAT64_C(  -447.11), EASYSIMD_FLOAT64_C(   946.74), EASYSIMD_FLOAT64_C(  -290.32), EASYSIMD_FLOAT64_C(   822.90),
        EASYSIMD_FLOAT64_C(  -341.83), EASYSIMD_FLOAT64_C(   394.17), EASYSIMD_FLOAT64_C(  -818.70), EASYSIMD_FLOAT64_C(  -927.38) },
      { EASYSIMD_FLOAT64_C(   304.96), EASYSIMD_FLOAT64_C(   632.45), EASYSIMD_FLOAT64_C(   537.34), EASYSIMD_FLOAT64_C(  -708.56),
        EASYSIMD_FLOAT64_C(  -862.18), EASYSIMD_FLOAT64_C(  -205.27), EASYSIMD_FLOAT64_C(   298.26), EASYSIMD_FLOAT64_C(   549.93) },
      {  INT64_C(  504077700776137226), -INT64_C( 4373231414786365212),  INT64_C( 1926090453807719414), -INT64_C( 3604477553545481946),
        -INT64_C( 6094970332675641500), -INT64_C( 7012434851621732680),  INT64_C( 2945758784224897206), -INT64_C( 6525066870547747102) },
       INT32_C(          61),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     1.57), EASYSIMD_FLOAT64_C(   537.34), EASYSIMD_FLOAT64_C(    90.00),
              -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(    90.00), EASYSIMD_FLOAT64_C(     1.00) } },
    { { EASYSIMD_FLOAT64_C(   -43.75), EASYSIMD_FLOAT64_C(   -12.63), EASYSIMD_FLOAT64_C(   241.36), EASYSIMD_FLOAT64_C(   746.59),
        EASYSIMD_FLOAT64_C(  -199.98), EASYSIMD_FLOAT64_C(    77.16), EASYSIMD_FLOAT64_C(  -140.10), EASYSIMD_FLOAT64_C(   824.72) },
      { EASYSIMD_FLOAT64_C(   682.50), EASYSIMD_FLOAT64_C(  -150.93), EASYSIMD_FLOAT64_C(  -991.81), EASYSIMD_FLOAT64_C(   767.05),
        EASYSIMD_FLOAT64_C(   536.99), EASYSIMD_FLOAT64_C(  -991.40), EASYSIMD_FLOAT64_C(  -275.01), EASYSIMD_FLOAT64_C(   104.98) },
      { -INT64_C( 7789806428759825058),  INT64_C( 3430688602923226970), -INT64_C( 8916928806116373579), -INT64_C( 4379815072206063591),
         INT64_C( 6129936730364833322),  INT64_C( 4650680796832418111),  INT64_C( 6113116675654518484),  INT64_C(  632811672515057823) },
       INT32_C(          33),
      {       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     1.00),
        EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(     1.00),             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   824.72) } },
    { { EASYSIMD_FLOAT64_C(  -879.52), EASYSIMD_FLOAT64_C(   263.63), EASYSIMD_FLOAT64_C(  -615.08), EASYSIMD_FLOAT64_C(   723.54),
        EASYSIMD_FLOAT64_C(  -623.64), EASYSIMD_FLOAT64_C(  -591.26), EASYSIMD_FLOAT64_C(  -598.61), EASYSIMD_FLOAT64_C(  -443.78) },
      { EASYSIMD_FLOAT64_C(   607.96), EASYSIMD_FLOAT64_C(   110.28), EASYSIMD_FLOAT64_C(  -209.40), EASYSIMD_FLOAT64_C(   223.93),
        EASYSIMD_FLOAT64_C(   596.81), EASYSIMD_FLOAT64_C(   879.52), EASYSIMD_FLOAT64_C(  -666.17), EASYSIMD_FLOAT64_C(  -279.15) },
      { -INT64_C( 2855587847951490119), -INT64_C( 6388242880835607013),  INT64_C(  884932204920817916),  INT64_C( 4520157740475609794),
        -INT64_C( 6013724346301306698), -INT64_C( 3623917963290514917), -INT64_C(  343134243671945207), -INT64_C( 8239661152358104093) },
       INT32_C(         127),
      {        EASYSIMD_MATH_INFINITY,             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(    90.00),             EASYSIMD_MATH_NAN,
        EASYSIMD_FLOAT64_C(     0.50), EASYSIMD_FLOAT64_C(  -591.26), EASYSIMD_FLOAT64_C(-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00), EASYSIMD_FLOAT64_C(     0.00) } },
  };

  easysimd__m512d a, b, r;
  easysimd__m512i c;

  a = easysimd_mm512_loadu_pd(test_vec[0].a);
  b = easysimd_mm512_loadu_pd(test_vec[0].b);
  c = easysimd_mm512_loadu_epi64(test_vec[0].c);
  r = easysimd_mm512_fixupimm_pd(a, b, c, INT32_C(         185));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[1].a);
  b = easysimd_mm512_loadu_pd(test_vec[1].b);
  c = easysimd_mm512_loadu_epi64(test_vec[1].c);
  r = easysimd_mm512_fixupimm_pd(a, b, c, INT32_C(          24));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[2].a);
  b = easysimd_mm512_loadu_pd(test_vec[2].b);
  c = easysimd_mm512_loadu_epi64(test_vec[2].c);
  r = easysimd_mm512_fixupimm_pd(a, b, c, INT32_C(         102));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[3].a);
  b = easysimd_mm512_loadu_pd(test_vec[3].b);
  c = easysimd_mm512_loadu_epi64(test_vec[3].c);
  r = easysimd_mm512_fixupimm_pd(a, b, c, INT32_C(         226));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[4].a);
  b = easysimd_mm512_loadu_pd(test_vec[4].b);
  c = easysimd_mm512_loadu_epi64(test_vec[4].c);
  r = easysimd_mm512_fixupimm_pd(a, b, c, INT32_C(         244));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[5].a);
  b = easysimd_mm512_loadu_pd(test_vec[5].b);
  c = easysimd_mm512_loadu_epi64(test_vec[5].c);
  r = easysimd_mm512_fixupimm_pd(a, b, c, INT32_C(          61));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[6].a);
  b = easysimd_mm512_loadu_pd(test_vec[6].b);
  c = easysimd_mm512_loadu_epi64(test_vec[6].c);
  r = easysimd_mm512_fixupimm_pd(a, b, c, INT32_C(          33));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[7].a);
  b = easysimd_mm512_loadu_pd(test_vec[7].b);
  c = easysimd_mm512_loadu_epi64(test_vec[7].c);
  r = easysimd_mm512_fixupimm_pd(a, b, c, INT32_C(         127));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float64 values[8 * 2 * sizeof(easysimd__m512d)];
  easysimd_test_x86_random_f64x8_full(8, 2, values, -1000.0, 1000.0, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512i c = easysimd_test_x86_random_i64x8();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m512d r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm512_fixupimm_pd, r, easysimd_mm512_setzero_pd(), imm8, a, b, c);

    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_fixupimm_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[8];
    const easysimd__mmask8 k;
    const easysimd_float64 b[8];
    const int64_t c[8];
    const int imm8;
    const easysimd_float64 r[8];
  } test_vec[] = {
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   426.54),             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   308.30),
        EASYSIMD_FLOAT64_C(   892.77), EASYSIMD_FLOAT64_C(   979.04), EASYSIMD_FLOAT64_C(   340.64), EASYSIMD_FLOAT64_C(   -39.81) },
      UINT8_C(128),
      { EASYSIMD_FLOAT64_C(   153.64),             EASYSIMD_MATH_NAN,             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -174.30),
        EASYSIMD_FLOAT64_C(   163.18), EASYSIMD_FLOAT64_C(    65.86), EASYSIMD_FLOAT64_C(   403.96), EASYSIMD_FLOAT64_C(    78.32) },
      {  INT64_C(  590413235852531120),  INT64_C( 7232391276447893876), -INT64_C(  888386226947612229), -INT64_C( 4944477350807124567),
         INT64_C( 1460193911303835525), -INT64_C( 1576120026583166417),  INT64_C(  793351936683446689),  INT64_C( 7393026991025107154) },
       INT32_C(          93),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   426.54),             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   308.30),
        EASYSIMD_FLOAT64_C(   892.77), EASYSIMD_FLOAT64_C(   979.04), EASYSIMD_FLOAT64_C(   340.64), EASYSIMD_FLOAT64_C(     1.57) } },
    { { EASYSIMD_FLOAT64_C(    36.69), EASYSIMD_FLOAT64_C(  -503.62), EASYSIMD_FLOAT64_C(   -93.23), EASYSIMD_FLOAT64_C(  -404.47),
        EASYSIMD_FLOAT64_C(   326.88), EASYSIMD_FLOAT64_C(  -754.64), EASYSIMD_FLOAT64_C(   621.89), EASYSIMD_FLOAT64_C(  -770.69) },
      UINT8_C(180),
      { EASYSIMD_FLOAT64_C(   814.93), EASYSIMD_FLOAT64_C(   763.48), EASYSIMD_FLOAT64_C(   823.78), EASYSIMD_FLOAT64_C(   -53.03),
        EASYSIMD_FLOAT64_C(   123.58), EASYSIMD_FLOAT64_C(   960.21), EASYSIMD_FLOAT64_C(  -392.78), EASYSIMD_FLOAT64_C(   -42.83) },
      { -INT64_C( 5501816326393228573), -INT64_C(   93621828897697856), -INT64_C(  398938361192923291),  INT64_C( 6920437769921158216),
         INT64_C( 5570524537727348327),  INT64_C( 7672535098681369098),  INT64_C( 4807441946526108433), -INT64_C( 3733867383390400686) },
       INT32_C(         145),
      { EASYSIMD_FLOAT64_C(    36.69), EASYSIMD_FLOAT64_C(  -503.62), EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(  -404.47),
        EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(   621.89), EASYSIMD_FLOAT64_C(179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00) } },
    { { EASYSIMD_FLOAT64_C(   386.75), EASYSIMD_FLOAT64_C(   524.19), EASYSIMD_FLOAT64_C(  -734.53), EASYSIMD_FLOAT64_C(   279.51),
        EASYSIMD_FLOAT64_C(   503.22), EASYSIMD_FLOAT64_C(   606.11), EASYSIMD_FLOAT64_C(  -760.30), EASYSIMD_FLOAT64_C(  -343.13) },
      UINT8_C(126),
      { EASYSIMD_FLOAT64_C(   657.20), EASYSIMD_FLOAT64_C(  -867.87), EASYSIMD_FLOAT64_C(   482.57), EASYSIMD_FLOAT64_C(  -179.62),
        EASYSIMD_FLOAT64_C(   197.99), EASYSIMD_FLOAT64_C(  -113.47), EASYSIMD_FLOAT64_C(   898.70), EASYSIMD_FLOAT64_C(  -765.33) },
      {  INT64_C( 6727256390144388277),  INT64_C( 4450448993850148734),  INT64_C( 4780392466732746210),  INT64_C( 4615893349345469324),
         INT64_C( 1857722319659112287),  INT64_C( 4809744760986285499),  INT64_C( 4074503859965546154), -INT64_C( 5730031931508629841) },
       INT32_C(         109),
      { EASYSIMD_FLOAT64_C(   386.75), EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00), EASYSIMD_FLOAT64_C(   279.51),
               EASYSIMD_MATH_INFINITY,             EASYSIMD_MATH_NAN,       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(  -343.13) } },
    { { EASYSIMD_FLOAT64_C(   382.91), EASYSIMD_FLOAT64_C(  -194.53), EASYSIMD_FLOAT64_C(  -169.80), EASYSIMD_FLOAT64_C(  -290.20),
        EASYSIMD_FLOAT64_C(    50.83), EASYSIMD_FLOAT64_C(  -547.91), EASYSIMD_FLOAT64_C(   -60.90), EASYSIMD_FLOAT64_C(  -134.24) },
      UINT8_C(  7),
      { EASYSIMD_FLOAT64_C(  -784.43), EASYSIMD_FLOAT64_C(  -237.12), EASYSIMD_FLOAT64_C(   812.73), EASYSIMD_FLOAT64_C(   339.16),
        EASYSIMD_FLOAT64_C(  -276.91), EASYSIMD_FLOAT64_C(  -580.05), EASYSIMD_FLOAT64_C(  -703.68), EASYSIMD_FLOAT64_C(  -890.17) },
      { -INT64_C( 6307778348516761119),  INT64_C( 8718859946731747332), -INT64_C( 3038133148536326361), -INT64_C( 9082269918958483748),
         INT64_C(   79109850622572653),  INT64_C( 3398901640614669510), -INT64_C( 8043556801939745910),  INT64_C( 6499667251476537873) },
       INT32_C(          44),
      {       -EASYSIMD_MATH_INFINITY,             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -290.20),
        EASYSIMD_FLOAT64_C(    50.83), EASYSIMD_FLOAT64_C(  -547.91), EASYSIMD_FLOAT64_C(   -60.90), EASYSIMD_FLOAT64_C(  -134.24) } },
    { { EASYSIMD_FLOAT64_C(   944.14), EASYSIMD_FLOAT64_C(  -438.21), EASYSIMD_FLOAT64_C(   389.35), EASYSIMD_FLOAT64_C(   447.36),
        EASYSIMD_FLOAT64_C(  -832.11), EASYSIMD_FLOAT64_C(   629.05), EASYSIMD_FLOAT64_C(  -895.77), EASYSIMD_FLOAT64_C(   825.09) },
      UINT8_C(216),
      { EASYSIMD_FLOAT64_C(   761.18), EASYSIMD_FLOAT64_C(   586.80), EASYSIMD_FLOAT64_C(  -354.53), EASYSIMD_FLOAT64_C(   -40.84),
        EASYSIMD_FLOAT64_C(  -526.67), EASYSIMD_FLOAT64_C(  -455.83), EASYSIMD_FLOAT64_C(   193.84), EASYSIMD_FLOAT64_C(   856.24) },
      { -INT64_C( 3217732326500979740), -INT64_C( 4141931513189136515),  INT64_C(  636567294509061446), -INT64_C( 2350493096095698755),
        -INT64_C( 3540806424903811191),  INT64_C( 7354415713565423886),  INT64_C( 3446419690301089847), -INT64_C( 1067888130130441900) },
       INT32_C(          24),
      { EASYSIMD_FLOAT64_C(   944.14), EASYSIMD_FLOAT64_C(  -438.21), EASYSIMD_FLOAT64_C(   389.35), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(   629.05), EASYSIMD_FLOAT64_C(    90.00),       -EASYSIMD_MATH_INFINITY } },
    { { EASYSIMD_FLOAT64_C(   349.64), EASYSIMD_FLOAT64_C(  -975.97), EASYSIMD_FLOAT64_C(  -433.96), EASYSIMD_FLOAT64_C(  -599.53),
        EASYSIMD_FLOAT64_C(  -523.87), EASYSIMD_FLOAT64_C(   505.14), EASYSIMD_FLOAT64_C(   266.23), EASYSIMD_FLOAT64_C(  -308.30) },
      UINT8_C(228),
      { EASYSIMD_FLOAT64_C(  -731.97), EASYSIMD_FLOAT64_C(    78.96), EASYSIMD_FLOAT64_C(  -969.15), EASYSIMD_FLOAT64_C(    -8.89),
        EASYSIMD_FLOAT64_C(   498.92), EASYSIMD_FLOAT64_C(  -672.82), EASYSIMD_FLOAT64_C(   100.95), EASYSIMD_FLOAT64_C(   443.06) },
      { -INT64_C(  562884167139706922),  INT64_C(  695238679323448907), -INT64_C( 6467260814660348465),  INT64_C(    7643986160136410),
        -INT64_C( 7769483103272170145),  INT64_C( 2793477892456537302), -INT64_C( 6380124243421683390), -INT64_C( 4928634145361480850) },
       INT32_C(           6),
      { EASYSIMD_FLOAT64_C(   349.64), EASYSIMD_FLOAT64_C(  -975.97), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -599.53),
        EASYSIMD_FLOAT64_C(  -523.87), EASYSIMD_FLOAT64_C(     0.50), EASYSIMD_FLOAT64_C(    90.00), EASYSIMD_FLOAT64_C(    -1.00) } },
    { { EASYSIMD_FLOAT64_C(  -111.04), EASYSIMD_FLOAT64_C(  -509.70), EASYSIMD_FLOAT64_C(  -109.58), EASYSIMD_FLOAT64_C(    56.86),
        EASYSIMD_FLOAT64_C(  -880.66), EASYSIMD_FLOAT64_C(    -5.35), EASYSIMD_FLOAT64_C(  -118.05), EASYSIMD_FLOAT64_C(   880.52) },
      UINT8_C(233),
      { EASYSIMD_FLOAT64_C(  -418.55), EASYSIMD_FLOAT64_C(   527.41), EASYSIMD_FLOAT64_C(  -160.32), EASYSIMD_FLOAT64_C(    54.78),
        EASYSIMD_FLOAT64_C(  -928.42), EASYSIMD_FLOAT64_C(  -966.48), EASYSIMD_FLOAT64_C(   -88.98), EASYSIMD_FLOAT64_C(   421.22) },
      {  INT64_C( 1716288905122210589),  INT64_C( 3760269252538005312), -INT64_C( 5896272160042592611), -INT64_C(  835412407286896683),
        -INT64_C( 2042030984283674393), -INT64_C( 4008075939584605642),  INT64_C( 1418288906546774858),  INT64_C( 5654826129669907373) },
       INT32_C(          34),
      { EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(  -509.70), EASYSIMD_FLOAT64_C(  -109.58), EASYSIMD_FLOAT64_C(    -0.00),
        EASYSIMD_FLOAT64_C(  -880.66), EASYSIMD_FLOAT64_C(    90.00), EASYSIMD_FLOAT64_C(-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { { EASYSIMD_FLOAT64_C(  -942.44), EASYSIMD_FLOAT64_C(   477.07), EASYSIMD_FLOAT64_C(   821.69), EASYSIMD_FLOAT64_C(  -466.32),
        EASYSIMD_FLOAT64_C(   -17.79), EASYSIMD_FLOAT64_C(    87.92), EASYSIMD_FLOAT64_C(   225.38), EASYSIMD_FLOAT64_C(   250.24) },
      UINT8_C( 38),
      { EASYSIMD_FLOAT64_C(  -833.12), EASYSIMD_FLOAT64_C(   256.23), EASYSIMD_FLOAT64_C(  -758.65), EASYSIMD_FLOAT64_C(   665.79),
        EASYSIMD_FLOAT64_C(   583.41), EASYSIMD_FLOAT64_C(   342.30), EASYSIMD_FLOAT64_C(   108.85), EASYSIMD_FLOAT64_C(  -527.63) },
      {  INT64_C( 7113024319111670152), -INT64_C( 2437665550224294166), -INT64_C( 3808900891834368166), -INT64_C( 2695208314107759990),
         INT64_C( 7542152230173275483),  INT64_C( 6813806618389862553), -INT64_C( 5378354398636598336),  INT64_C( 6331146260969900465) },
       INT32_C(         180),
      { EASYSIMD_FLOAT64_C(  -942.44), EASYSIMD_FLOAT64_C(179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00), EASYSIMD_FLOAT64_C(     1.57), EASYSIMD_FLOAT64_C(  -466.32),
        EASYSIMD_FLOAT64_C(   -17.79), EASYSIMD_FLOAT64_C(     0.50), EASYSIMD_FLOAT64_C(   225.38), EASYSIMD_FLOAT64_C(   250.24) } },
  };

  easysimd__m512d a, b, r;
  easysimd__m512i c;

  a = easysimd_mm512_loadu_pd(test_vec[0].a);
  b = easysimd_mm512_loadu_pd(test_vec[0].b);
  c = easysimd_mm512_loadu_epi64(test_vec[0].c);
  r = easysimd_mm512_mask_fixupimm_pd(a, test_vec[0].k, b, c, INT32_C(          93));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[1].a);
  b = easysimd_mm512_loadu_pd(test_vec[1].b);
  c = easysimd_mm512_loadu_epi64(test_vec[1].c);
  r = easysimd_mm512_mask_fixupimm_pd(a, test_vec[1].k, b, c, INT32_C(         145));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[2].a);
  b = easysimd_mm512_loadu_pd(test_vec[2].b);
  c = easysimd_mm512_loadu_epi64(test_vec[2].c);
  r = easysimd_mm512_mask_fixupimm_pd(a, test_vec[2].k, b, c, INT32_C(         109));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[3].a);
  b = easysimd_mm512_loadu_pd(test_vec[3].b);
  c = easysimd_mm512_loadu_epi64(test_vec[3].c);
  r = easysimd_mm512_mask_fixupimm_pd(a, test_vec[3].k, b, c, INT32_C(          44));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[4].a);
  b = easysimd_mm512_loadu_pd(test_vec[4].b);
  c = easysimd_mm512_loadu_epi64(test_vec[4].c);
  r = easysimd_mm512_mask_fixupimm_pd(a, test_vec[4].k, b, c, INT32_C(          24));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[5].a);
  b = easysimd_mm512_loadu_pd(test_vec[5].b);
  c = easysimd_mm512_loadu_epi64(test_vec[5].c);
  r = easysimd_mm512_mask_fixupimm_pd(a, test_vec[5].k, b, c, INT32_C(           6));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[6].a);
  b = easysimd_mm512_loadu_pd(test_vec[6].b);
  c = easysimd_mm512_loadu_epi64(test_vec[6].c);
  r = easysimd_mm512_mask_fixupimm_pd(a, test_vec[6].k, b, c, INT32_C(          34));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[7].a);
  b = easysimd_mm512_loadu_pd(test_vec[7].b);
  c = easysimd_mm512_loadu_epi64(test_vec[7].c);
  r = easysimd_mm512_mask_fixupimm_pd(a, test_vec[7].k, b, c, INT32_C(         180));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float64 values[8 * 2 * sizeof(easysimd__m512d)];
  easysimd_test_x86_random_f64x8_full(8, 2, values, -1000.0, 1000.0, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m512d a = easysimd_test_x86_random_extract_f64x8(i, 2, 0, values);
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512d b = easysimd_test_x86_random_extract_f64x8(i, 2, 1, values);
    easysimd__m512i c = easysimd_test_x86_random_i64x8();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m512d r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm512_mask_fixupimm_pd, r, easysimd_mm512_setzero_pd(), imm8, a, k, b, c);

    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_fixupimm_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const int64_t c[8];
    const int imm8;
    const easysimd_float64 r[8];
  } test_vec[] = {
    { UINT8_C(  7),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   955.79),             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -670.42),
        EASYSIMD_FLOAT64_C(   507.78), EASYSIMD_FLOAT64_C(   668.53), EASYSIMD_FLOAT64_C(  -312.88), EASYSIMD_FLOAT64_C(  -676.68) },
      { EASYSIMD_FLOAT64_C(  -788.81),             EASYSIMD_MATH_NAN,             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   787.75),
        EASYSIMD_FLOAT64_C(   -59.54), EASYSIMD_FLOAT64_C(  -201.30), EASYSIMD_FLOAT64_C(  -215.65), EASYSIMD_FLOAT64_C(  -674.20) },
      {  INT64_C( 1980430139847174865),  INT64_C( 7759446902478975134),  INT64_C( 3439327106012688852),  INT64_C( 4195722529781328997),
        -INT64_C( 3498767833470973027),  INT64_C( 2365852478743917713),  INT64_C( 7951859446474610787), -INT64_C( 2053963285188767013) },
       INT32_C(          38),
      { EASYSIMD_FLOAT64_C(    90.00), EASYSIMD_FLOAT64_C(179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(197),
      { EASYSIMD_FLOAT64_C(  -210.67), EASYSIMD_FLOAT64_C(  -923.92), EASYSIMD_FLOAT64_C(  -266.07), EASYSIMD_FLOAT64_C(  -544.13),
        EASYSIMD_FLOAT64_C(  -109.49), EASYSIMD_FLOAT64_C(   289.78), EASYSIMD_FLOAT64_C(   695.40), EASYSIMD_FLOAT64_C(   759.47) },
      { EASYSIMD_FLOAT64_C(   100.95), EASYSIMD_FLOAT64_C(   466.46), EASYSIMD_FLOAT64_C(   -23.57), EASYSIMD_FLOAT64_C(   821.39),
        EASYSIMD_FLOAT64_C(   316.27), EASYSIMD_FLOAT64_C(   446.82), EASYSIMD_FLOAT64_C(  -962.93), EASYSIMD_FLOAT64_C(  -184.53) },
      { -INT64_C( 2582038496707926102), -INT64_C( 1317467548538680682), -INT64_C( 6837576086108756540), -INT64_C( 2003221978205912950),
        -INT64_C( 2103764540477699513),  INT64_C( 4279343748859563320), -INT64_C( 4479623439718470803),  INT64_C(  381508258574081326) },
       INT32_C(         237),
      { EASYSIMD_FLOAT64_C(     1.57), EASYSIMD_FLOAT64_C(     0.00),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(    -0.00) } },
    { UINT8_C(252),
      { EASYSIMD_FLOAT64_C(   402.61), EASYSIMD_FLOAT64_C(  -787.53), EASYSIMD_FLOAT64_C(   145.06), EASYSIMD_FLOAT64_C(   -89.61),
        EASYSIMD_FLOAT64_C(   881.00), EASYSIMD_FLOAT64_C(   832.17), EASYSIMD_FLOAT64_C(   233.71), EASYSIMD_FLOAT64_C(  -907.81) },
      { EASYSIMD_FLOAT64_C(   559.55), EASYSIMD_FLOAT64_C(  -338.73), EASYSIMD_FLOAT64_C(   879.94), EASYSIMD_FLOAT64_C(  -499.99),
        EASYSIMD_FLOAT64_C(   459.97), EASYSIMD_FLOAT64_C(  -335.71), EASYSIMD_FLOAT64_C(  -174.18), EASYSIMD_FLOAT64_C(  -750.70) },
      { -INT64_C(  916294869243913276), -INT64_C( 4435599694132837618), -INT64_C( 4195850299745750422), -INT64_C( 8813848203558204765),
        -INT64_C( 7303256893586742778),  INT64_C(  466199096505665057),  INT64_C( 7250465275341858935),  INT64_C( 4892250858803581342) },
       INT32_C(         166),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   145.06),             EASYSIMD_MATH_NAN,
        EASYSIMD_FLOAT64_C(     0.50),        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(    -1.00), EASYSIMD_FLOAT64_C(-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00) } },
    { UINT8_C(216),
      { EASYSIMD_FLOAT64_C(  -259.63), EASYSIMD_FLOAT64_C(   559.75), EASYSIMD_FLOAT64_C(  -294.83), EASYSIMD_FLOAT64_C(   630.88),
        EASYSIMD_FLOAT64_C(  -150.47), EASYSIMD_FLOAT64_C(  -599.43), EASYSIMD_FLOAT64_C(   390.35), EASYSIMD_FLOAT64_C(   950.48) },
      { EASYSIMD_FLOAT64_C(   867.03), EASYSIMD_FLOAT64_C(  -633.21), EASYSIMD_FLOAT64_C(   771.87), EASYSIMD_FLOAT64_C(   183.30),
        EASYSIMD_FLOAT64_C(   813.61), EASYSIMD_FLOAT64_C(   808.94), EASYSIMD_FLOAT64_C(   998.77), EASYSIMD_FLOAT64_C(   216.22) },
      { -INT64_C( 1856416937153016422),  INT64_C( 2616370727667712550), -INT64_C( 8258720143844920363),  INT64_C( 3825639225440839173),
        -INT64_C( 1893178983020359629),  INT64_C( 2651146073228449621), -INT64_C( 5562317530136408969), -INT64_C( 7339732250223640237) },
       INT32_C(          24),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),             EASYSIMD_MATH_NAN,
        EASYSIMD_FLOAT64_C(     1.57), EASYSIMD_FLOAT64_C(     0.00),       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00) } },
    { UINT8_C( 13),
      { EASYSIMD_FLOAT64_C(  -978.59), EASYSIMD_FLOAT64_C(   143.82), EASYSIMD_FLOAT64_C(  -873.39), EASYSIMD_FLOAT64_C(   902.41),
        EASYSIMD_FLOAT64_C(   -24.00), EASYSIMD_FLOAT64_C(   360.31), EASYSIMD_FLOAT64_C(   994.60), EASYSIMD_FLOAT64_C(  -464.45) },
      { EASYSIMD_FLOAT64_C(  -978.42), EASYSIMD_FLOAT64_C(   874.53), EASYSIMD_FLOAT64_C(    35.56), EASYSIMD_FLOAT64_C(   481.55),
        EASYSIMD_FLOAT64_C(  -461.18), EASYSIMD_FLOAT64_C(   861.38), EASYSIMD_FLOAT64_C(   730.85), EASYSIMD_FLOAT64_C(   279.19) },
      { -INT64_C(  896769177251759320), -INT64_C( 6193387382336998701), -INT64_C( 5196612256581894536),  INT64_C( 1376032110800763920),
         INT64_C( 6306131862722149217),  INT64_C( 1169742488768277908),  INT64_C( 3955968233953704320), -INT64_C( 4835359274496313880) },
       INT32_C(         213),
      { EASYSIMD_FLOAT64_C(  -978.59), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    -0.00), EASYSIMD_FLOAT64_C(   902.41),
        EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 26),
      { EASYSIMD_FLOAT64_C(   421.12), EASYSIMD_FLOAT64_C(  -563.98), EASYSIMD_FLOAT64_C(   -89.93), EASYSIMD_FLOAT64_C(  -729.35),
        EASYSIMD_FLOAT64_C(  -163.42), EASYSIMD_FLOAT64_C(  -699.58), EASYSIMD_FLOAT64_C(  -778.87), EASYSIMD_FLOAT64_C(  -296.39) },
      { EASYSIMD_FLOAT64_C(  -332.80), EASYSIMD_FLOAT64_C(   993.00), EASYSIMD_FLOAT64_C(   886.91), EASYSIMD_FLOAT64_C(  -519.19),
        EASYSIMD_FLOAT64_C(   801.94), EASYSIMD_FLOAT64_C(   885.68), EASYSIMD_FLOAT64_C(   697.03), EASYSIMD_FLOAT64_C(   823.35) },
      {  INT64_C( 3050414041704870352),  INT64_C( 6690900923770431098), -INT64_C( 5993983986114490175), -INT64_C( 6174474833854624260),
        -INT64_C( 5323328165015716467),  INT64_C( 7381475265046301219), -INT64_C( 4011670653339126699),  INT64_C( 4449359276032250939) },
       INT32_C(          73),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.50), EASYSIMD_FLOAT64_C(     0.00),        EASYSIMD_MATH_INFINITY,
        EASYSIMD_FLOAT64_C(-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(228),
      { EASYSIMD_FLOAT64_C(    29.50), EASYSIMD_FLOAT64_C(   823.63), EASYSIMD_FLOAT64_C(   725.76), EASYSIMD_FLOAT64_C(  -994.50),
        EASYSIMD_FLOAT64_C(   183.95), EASYSIMD_FLOAT64_C(   720.35), EASYSIMD_FLOAT64_C(  -458.96), EASYSIMD_FLOAT64_C(   205.53) },
      { EASYSIMD_FLOAT64_C(   594.89), EASYSIMD_FLOAT64_C(   576.61), EASYSIMD_FLOAT64_C(  -312.92), EASYSIMD_FLOAT64_C(  -866.29),
        EASYSIMD_FLOAT64_C(   437.98), EASYSIMD_FLOAT64_C(  -582.07), EASYSIMD_FLOAT64_C(   412.90), EASYSIMD_FLOAT64_C(  -140.89) },
      { -INT64_C( 1530913097592128687), -INT64_C( 5689482277608806518),  INT64_C( 1999038372931031609),  INT64_C( 9137215602403113922),
         INT64_C( 8770016930550174757),  INT64_C( 1528904846629467522), -INT64_C( 1621048306553379691), -INT64_C(  898573893360229156) },
       INT32_C(         115),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00), EASYSIMD_FLOAT64_C(     0.00),
        EASYSIMD_FLOAT64_C(     0.00),       -EASYSIMD_MATH_INFINITY,        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(    -1.00) } },
    { UINT8_C(125),
      { EASYSIMD_FLOAT64_C(  -146.06), EASYSIMD_FLOAT64_C(  -677.03), EASYSIMD_FLOAT64_C(   129.76), EASYSIMD_FLOAT64_C(   690.52),
        EASYSIMD_FLOAT64_C(  -376.61), EASYSIMD_FLOAT64_C(   350.89), EASYSIMD_FLOAT64_C(  -605.87), EASYSIMD_FLOAT64_C(   290.59) },
      { EASYSIMD_FLOAT64_C(   343.89), EASYSIMD_FLOAT64_C(  -718.96), EASYSIMD_FLOAT64_C(   771.41), EASYSIMD_FLOAT64_C(   145.84),
        EASYSIMD_FLOAT64_C(  -833.29), EASYSIMD_FLOAT64_C(   468.43), EASYSIMD_FLOAT64_C(   -30.81), EASYSIMD_FLOAT64_C(   196.21) },
      {  INT64_C( 6144183755008432755),  INT64_C( 6703975818607614397),  INT64_C( 7170085494338482894), -INT64_C( 6527625493052230455),
        -INT64_C( 5982659440781924078),  INT64_C(   37320938593641387), -INT64_C( 3985627399097959031), -INT64_C( 3422703011251321176) },
       INT32_C(         134),
      {        EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.50), EASYSIMD_FLOAT64_C(179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00),
              -EASYSIMD_MATH_INFINITY,             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(     1.57), EASYSIMD_FLOAT64_C(     0.00) } },
  };

  easysimd__m512d a, b, r;
  easysimd__m512i c;

  a = easysimd_mm512_loadu_pd(test_vec[0].a);
  b = easysimd_mm512_loadu_pd(test_vec[0].b);
  c = easysimd_mm512_loadu_epi64(test_vec[0].c);
  r = easysimd_mm512_maskz_fixupimm_pd(test_vec[0].k, a, b, c, INT32_C(          38));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[1].a);
  b = easysimd_mm512_loadu_pd(test_vec[1].b);
  c = easysimd_mm512_loadu_epi64(test_vec[1].c);
  r = easysimd_mm512_maskz_fixupimm_pd(test_vec[1].k, a, b, c, INT32_C(         237));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[2].a);
  b = easysimd_mm512_loadu_pd(test_vec[2].b);
  c = easysimd_mm512_loadu_epi64(test_vec[2].c);
  r = easysimd_mm512_maskz_fixupimm_pd(test_vec[2].k, a, b, c, INT32_C(         166));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[3].a);
  b = easysimd_mm512_loadu_pd(test_vec[3].b);
  c = easysimd_mm512_loadu_epi64(test_vec[3].c);
  r = easysimd_mm512_maskz_fixupimm_pd(test_vec[3].k, a, b, c, INT32_C(          24));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[4].a);
  b = easysimd_mm512_loadu_pd(test_vec[4].b);
  c = easysimd_mm512_loadu_epi64(test_vec[4].c);
  r = easysimd_mm512_maskz_fixupimm_pd(test_vec[4].k, a, b, c, INT32_C(         213));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[5].a);
  b = easysimd_mm512_loadu_pd(test_vec[5].b);
  c = easysimd_mm512_loadu_epi64(test_vec[5].c);
  r = easysimd_mm512_maskz_fixupimm_pd(test_vec[5].k, a, b, c, INT32_C(          73));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[6].a);
  b = easysimd_mm512_loadu_pd(test_vec[6].b);
  c = easysimd_mm512_loadu_epi64(test_vec[6].c);
  r = easysimd_mm512_maskz_fixupimm_pd(test_vec[6].k, a, b, c, INT32_C(         115));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[7].a);
  b = easysimd_mm512_loadu_pd(test_vec[7].b);
  c = easysimd_mm512_loadu_epi64(test_vec[7].c);
  r = easysimd_mm512_maskz_fixupimm_pd(test_vec[7].k, a, b, c, INT32_C(         134));
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float64 values[8 * 2 * sizeof(easysimd__m512d)];
  easysimd_test_x86_random_f64x8_full(8, 2, values, -1000.0, 1000.0, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512d a = easysimd_test_x86_random_extract_f64x8(i, 2, 0, values);
    easysimd__m512d b = easysimd_test_x86_random_extract_f64x8(i, 2, 1, values);
    easysimd__m512i c = easysimd_test_x86_random_i64x8();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m512d r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm512_maskz_fixupimm_pd, r, easysimd_mm512_setzero_pd(), imm8, k, a, b, c);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_fixupimm_sd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const int64_t c[2];
    const int imm8;
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -434.65), EASYSIMD_FLOAT64_C(   962.35) },
      { EASYSIMD_FLOAT64_C(   993.85), EASYSIMD_FLOAT64_C(   464.97) },
      { -INT64_C( 6201712406494312261), -INT64_C( 6874643461497553272) },
       INT32_C(         203),
      { EASYSIMD_FLOAT64_C(     1.00), EASYSIMD_FLOAT64_C(   464.97) } },
    { { EASYSIMD_FLOAT64_C(  -229.92), EASYSIMD_FLOAT64_C(   778.14) },
      { EASYSIMD_FLOAT64_C(   766.55), EASYSIMD_FLOAT64_C(   267.74) },
      { -INT64_C( 5231967822968100878), -INT64_C( 3932124877831919016) },
       INT32_C(          71),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   267.74) } },
    { { EASYSIMD_FLOAT64_C(  -361.64), EASYSIMD_FLOAT64_C(   175.70) },
      { EASYSIMD_FLOAT64_C(  -916.41), EASYSIMD_FLOAT64_C(  -413.48) },
      { -INT64_C( 6365964835022372157),  INT64_C(  407364307603054067) },
       INT32_C(         105),
      { EASYSIMD_FLOAT64_C(     0.50), EASYSIMD_FLOAT64_C(  -413.48) } },
    { { EASYSIMD_FLOAT64_C(  -283.55), EASYSIMD_FLOAT64_C(  -203.94) },
      { EASYSIMD_FLOAT64_C(  -182.37), EASYSIMD_FLOAT64_C(   653.39) },
      {  INT64_C( 5905104829890180511),  INT64_C( 5729690731103965223) },
       INT32_C(          80),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   653.39) } },
    { { EASYSIMD_FLOAT64_C(  -970.08), EASYSIMD_FLOAT64_C(  -756.16) },
      { EASYSIMD_FLOAT64_C(  -203.03), EASYSIMD_FLOAT64_C(   763.63) },
      { -INT64_C( 6573244426834817559),  INT64_C( 2250106535311627098) },
       INT32_C(         203),
      {       -EASYSIMD_MATH_INFINITY, EASYSIMD_FLOAT64_C(   763.63) } },
    { { EASYSIMD_FLOAT64_C(   726.84), EASYSIMD_FLOAT64_C(   884.76) },
      { EASYSIMD_FLOAT64_C(  -723.31), EASYSIMD_FLOAT64_C(   418.97) },
      { -INT64_C( 8755337847876634916), -INT64_C( 5818225939372691561) },
       INT32_C(          85),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   418.97) } },
    { { EASYSIMD_FLOAT64_C(  -746.17), EASYSIMD_FLOAT64_C(   635.84) },
      { EASYSIMD_FLOAT64_C(  -144.68), EASYSIMD_FLOAT64_C(   455.53) },
      { -INT64_C( 4351694503276025513), -INT64_C( 4285814763654716830) },
       INT32_C(         234),
      { EASYSIMD_FLOAT64_C(  -144.68), EASYSIMD_FLOAT64_C(   455.53) } },
    { { EASYSIMD_FLOAT64_C(   463.48), EASYSIMD_FLOAT64_C(   415.58) },
      { EASYSIMD_FLOAT64_C(    45.32), EASYSIMD_FLOAT64_C(  -277.94) },
      { -INT64_C( 5818981292810336533), -INT64_C( 8942148710272306116) },
       INT32_C(         126),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -277.94) } },
  };

  easysimd__m128d a, b, r;
  easysimd__m128i c;

  a = easysimd_mm_loadu_pd(test_vec[0].a);
  b = easysimd_mm_loadu_pd(test_vec[0].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[0].c);
  r = easysimd_mm_fixupimm_sd(a, b, c, INT32_C(         203));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[1].a);
  b = easysimd_mm_loadu_pd(test_vec[1].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[1].c);
  r = easysimd_mm_fixupimm_sd(a, b, c, INT32_C(          71));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[2].a);
  b = easysimd_mm_loadu_pd(test_vec[2].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[2].c);
  r = easysimd_mm_fixupimm_sd(a, b, c, INT32_C(         105));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[3].a);
  b = easysimd_mm_loadu_pd(test_vec[3].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[3].c);
  r = easysimd_mm_fixupimm_sd(a, b, c, INT32_C(          80));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[4].a);
  b = easysimd_mm_loadu_pd(test_vec[4].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[4].c);
  r = easysimd_mm_fixupimm_sd(a, b, c, INT32_C(         203));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[5].a);
  b = easysimd_mm_loadu_pd(test_vec[5].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[5].c);
  r = easysimd_mm_fixupimm_sd(a, b, c, INT32_C(          85));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[6].a);
  b = easysimd_mm_loadu_pd(test_vec[6].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[6].c);
  r = easysimd_mm_fixupimm_sd(a, b, c, INT32_C(         234));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[7].a);
  b = easysimd_mm_loadu_pd(test_vec[7].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[7].c);
  r = easysimd_mm_fixupimm_sd(a, b, c, INT32_C(         126));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float64 values[8 * 2 * sizeof(easysimd__m128d)];
  easysimd_test_x86_random_f64x2_full(8, 2, values, -1000.0f, 1000.0f, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128i c = easysimd_test_x86_random_i64x2();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m128d r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm_fixupimm_sd, r, easysimd_mm_setzero_pd(), imm8, a, b, c);

    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_fixupimm_sd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[2];
    const easysimd__mmask8 k;
    const easysimd_float64 b[2];
    const int64_t c[2];
    const int imm8;
    const easysimd_float64 r[2];
  } test_vec[] = {
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -946.07) },
      UINT8_C(108),
      { EASYSIMD_FLOAT64_C(   350.22),             EASYSIMD_MATH_NAN },
      {  INT64_C( 7276337534214629559), -INT64_C( 5782142652990400288) },
       INT32_C(          54),
      {             EASYSIMD_MATH_NAN,             EASYSIMD_MATH_NAN } },
    { {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   -87.44) },
      UINT8_C( 16),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   113.43) },
      {  INT64_C( 4437018654289254476),  INT64_C( 8281413503981846293) },
       INT32_C(         119),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   113.43) } },
    { { EASYSIMD_FLOAT64_C(  -671.86), EASYSIMD_FLOAT64_C(  -634.23) },
      UINT8_C( 48),
      { EASYSIMD_FLOAT64_C(   835.49), EASYSIMD_FLOAT64_C(  -515.97) },
      {  INT64_C( 5823463708220961977),  INT64_C(  549944464615793235) },
       INT32_C(          94),
      { EASYSIMD_FLOAT64_C(  -671.86), EASYSIMD_FLOAT64_C(  -515.97) } },
    { { EASYSIMD_FLOAT64_C(    96.49), EASYSIMD_FLOAT64_C(  -849.16) },
      UINT8_C(211),
      { EASYSIMD_FLOAT64_C(  -106.25), EASYSIMD_FLOAT64_C(   463.04) },
      { -INT64_C( 2893060814060645733),  INT64_C(  481335323978107287) },
       INT32_C(         220),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   463.04) } },
    { { EASYSIMD_FLOAT64_C(   321.50), EASYSIMD_FLOAT64_C(  -240.99) },
      UINT8_C(156),
      { EASYSIMD_FLOAT64_C(  -345.61), EASYSIMD_FLOAT64_C(   298.77) },
      {  INT64_C( 8540168881747480837),  INT64_C( 8122973825656772644) },
       INT32_C(         114),
      { EASYSIMD_FLOAT64_C(   321.50), EASYSIMD_FLOAT64_C(   298.77) } },
    { { EASYSIMD_FLOAT64_C(   797.95), EASYSIMD_FLOAT64_C(  -602.64) },
      UINT8_C( 34),
      { EASYSIMD_FLOAT64_C(   967.39), EASYSIMD_FLOAT64_C(  -712.73) },
      { -INT64_C( 4034933418764703661), -INT64_C( 3858895469253124664) },
       INT32_C(         248),
      { EASYSIMD_FLOAT64_C(   797.95), EASYSIMD_FLOAT64_C(  -712.73) } },
    { { EASYSIMD_FLOAT64_C(  -889.05), EASYSIMD_FLOAT64_C(  -562.50) },
      UINT8_C(249),
      { EASYSIMD_FLOAT64_C(  -719.59), EASYSIMD_FLOAT64_C(   909.21) },
      {  INT64_C( 8174038176446478372), -INT64_C( 3938460821565959552) },
       INT32_C(         199),
      { EASYSIMD_FLOAT64_C(-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00), EASYSIMD_FLOAT64_C(   909.21) } },
    { { EASYSIMD_FLOAT64_C(   849.16), EASYSIMD_FLOAT64_C(  -925.34) },
      UINT8_C(237),
      { EASYSIMD_FLOAT64_C(  -296.82), EASYSIMD_FLOAT64_C(   106.07) },
      {  INT64_C( 2451463648244230570), -INT64_C( 7635248910846984279) },
       INT32_C(         150),
      { EASYSIMD_FLOAT64_C(     1.57), EASYSIMD_FLOAT64_C(   106.07) } },
  };

  easysimd__m128d a, b, r;
  easysimd__m128i c;

  a = easysimd_mm_loadu_pd(test_vec[0].a);
  b = easysimd_mm_loadu_pd(test_vec[0].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[0].c);
  r = easysimd_mm_mask_fixupimm_sd(a, test_vec[0].k, b, c, INT32_C(          54));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[1].a);
  b = easysimd_mm_loadu_pd(test_vec[1].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[1].c);
  r = easysimd_mm_mask_fixupimm_sd(a, test_vec[1].k, b, c, INT32_C(         119));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[2].a);
  b = easysimd_mm_loadu_pd(test_vec[2].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[2].c);
  r = easysimd_mm_mask_fixupimm_sd(a, test_vec[2].k, b, c, INT32_C(          94));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[3].a);
  b = easysimd_mm_loadu_pd(test_vec[3].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[3].c);
  r = easysimd_mm_mask_fixupimm_sd(a, test_vec[3].k, b, c, INT32_C(         220));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[4].a);
  b = easysimd_mm_loadu_pd(test_vec[4].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[4].c);
  r = easysimd_mm_mask_fixupimm_sd(a, test_vec[4].k, b, c, INT32_C(         114));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[5].a);
  b = easysimd_mm_loadu_pd(test_vec[5].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[5].c);
  r = easysimd_mm_mask_fixupimm_sd(a, test_vec[5].k, b, c, INT32_C(         248));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[6].a);
  b = easysimd_mm_loadu_pd(test_vec[6].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[6].c);
  r = easysimd_mm_mask_fixupimm_sd(a, test_vec[6].k, b, c, INT32_C(         199));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[7].a);
  b = easysimd_mm_loadu_pd(test_vec[7].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[7].c);
  r = easysimd_mm_mask_fixupimm_sd(a, test_vec[7].k, b, c, INT32_C(         150));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float64 values[8 * 2 * sizeof(easysimd__m128d)];
  easysimd_test_x86_random_f64x2_full(8, 2, values, -1000.0, 1000.0, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__m128d a = easysimd_test_x86_random_extract_f64x2(i, 2, 0, values);
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d b = easysimd_test_x86_random_extract_f64x2(i, 2, 1, values);
    easysimd__m128i c = easysimd_test_x86_random_i64x2();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m128d r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm_mask_fixupimm_sd, r, easysimd_mm_setzero_pd(), imm8, a, k, b, c);

    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_fixupimm_sd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const int64_t c[2];
    const int imm8;
    const easysimd_float64 r[2];
  } test_vec[] = {
    { UINT8_C( 74),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   749.77) },
      { EASYSIMD_FLOAT64_C(  -164.18),             EASYSIMD_MATH_NAN },
      {  INT64_C( 3888319529157145603), -INT64_C(  334497910293994631) },
       INT32_C(         144),
      { EASYSIMD_FLOAT64_C(     0.00),             EASYSIMD_MATH_NAN } },
    { UINT8_C(138),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -791.31) },
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(   -29.79) },
      {  INT64_C( 3687016637497353641),  INT64_C( 3048913945180468248) },
       INT32_C(         226),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   -29.79) } },
    { UINT8_C( 85),
      { EASYSIMD_FLOAT64_C(   211.27), EASYSIMD_FLOAT64_C(  -689.48) },
      { EASYSIMD_FLOAT64_C(   804.79), EASYSIMD_FLOAT64_C(  -849.39) },
      { -INT64_C( 5305374856903872031), -INT64_C( 3419851811847149235) },
       INT32_C(         104),
      {             EASYSIMD_MATH_NAN, EASYSIMD_FLOAT64_C(  -849.39) } },
    { UINT8_C(244),
      { EASYSIMD_FLOAT64_C(  -448.53), EASYSIMD_FLOAT64_C(  -269.00) },
      { EASYSIMD_FLOAT64_C(  -569.85), EASYSIMD_FLOAT64_C(  -410.93) },
      { -INT64_C( 4825112106579726283), -INT64_C( 2817603741860012409) },
       INT32_C(          59),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -410.93) } },
    { UINT8_C(251),
      { EASYSIMD_FLOAT64_C(   -45.20), EASYSIMD_FLOAT64_C(   408.40) },
      { EASYSIMD_FLOAT64_C(  -979.11), EASYSIMD_FLOAT64_C(   147.70) },
      { -INT64_C( 4404905431584405097),  INT64_C( 8468874734379455357) },
       INT32_C(         134),
      { EASYSIMD_FLOAT64_C(  -979.11), EASYSIMD_FLOAT64_C(   147.70) } },
    { UINT8_C(228),
      { EASYSIMD_FLOAT64_C(   580.60), EASYSIMD_FLOAT64_C(   173.27) },
      { EASYSIMD_FLOAT64_C(  -190.14), EASYSIMD_FLOAT64_C(   403.84) },
      {  INT64_C( 3798499378025477940), -INT64_C( 8357857463700311777) },
       INT32_C(         203),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   403.84) } },
    { UINT8_C(187),
      { EASYSIMD_FLOAT64_C(   771.60), EASYSIMD_FLOAT64_C(   -72.57) },
      { EASYSIMD_FLOAT64_C(  -868.59), EASYSIMD_FLOAT64_C(  -661.99) },
      { -INT64_C( 4448220770040524425), -INT64_C( 2024067965234304862) },
       INT32_C(         198),
      { EASYSIMD_FLOAT64_C(-179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.00), EASYSIMD_FLOAT64_C(  -661.99) } },
    { UINT8_C(233),
      { EASYSIMD_FLOAT64_C(  -592.15), EASYSIMD_FLOAT64_C(   752.81) },
      { EASYSIMD_FLOAT64_C(  -532.61), EASYSIMD_FLOAT64_C(   187.17) },
      { -INT64_C(  436671115233434838), -INT64_C( 6749082864589886401) },
       INT32_C(         180),
      { EASYSIMD_FLOAT64_C(    90.00), EASYSIMD_FLOAT64_C(   187.17) } },
  };

  easysimd__m128d a, b, r;
  easysimd__m128i c;

  a = easysimd_mm_loadu_pd(test_vec[0].a);
  b = easysimd_mm_loadu_pd(test_vec[0].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[0].c);
  r = easysimd_mm_maskz_fixupimm_sd(test_vec[0].k, a, b, c, INT32_C(         144));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[1].a);
  b = easysimd_mm_loadu_pd(test_vec[1].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[1].c);
  r = easysimd_mm_maskz_fixupimm_sd(test_vec[1].k, a, b, c, INT32_C(         226));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[2].a);
  b = easysimd_mm_loadu_pd(test_vec[2].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[2].c);
  r = easysimd_mm_maskz_fixupimm_sd(test_vec[2].k, a, b, c, INT32_C(         104));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[3].a);
  b = easysimd_mm_loadu_pd(test_vec[3].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[3].c);
  r = easysimd_mm_maskz_fixupimm_sd(test_vec[3].k, a, b, c, INT32_C(          59));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[4].a);
  b = easysimd_mm_loadu_pd(test_vec[4].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[4].c);
  r = easysimd_mm_maskz_fixupimm_sd(test_vec[4].k, a, b, c, INT32_C(         134));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[5].a);
  b = easysimd_mm_loadu_pd(test_vec[5].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[5].c);
  r = easysimd_mm_maskz_fixupimm_sd(test_vec[5].k, a, b, c, INT32_C(         203));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[6].a);
  b = easysimd_mm_loadu_pd(test_vec[6].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[6].c);
  r = easysimd_mm_maskz_fixupimm_sd(test_vec[6].k, a, b, c, INT32_C(         198));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[7].a);
  b = easysimd_mm_loadu_pd(test_vec[7].b);
  c = easysimd_x_mm_loadu_epi64(test_vec[7].c);
  r = easysimd_mm_maskz_fixupimm_sd(test_vec[7].k, a, b, c, INT32_C(         180));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  easysimd_float64 values[8 * 2 * sizeof(easysimd__m128d)];
  easysimd_test_x86_random_f64x2_full(8, 2, values, -1000.0, 1000.0, EASYSIMD_TEST_VEC_FLOAT_NAN);

  for (size_t i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_extract_f64x2(i, 2, 0, values);
    easysimd__m128d b = easysimd_test_x86_random_extract_f64x2(i, 2, 1, values);
    easysimd__m128i c = easysimd_test_x86_random_i64x2();
    int32_t imm8 = easysimd_test_codegen_random_i32() & 255;
    easysimd__m128d r;
    EASYSIMD_CONSTIFY_256_(easysimd_mm_maskz_fixupimm_sd, r, easysimd_mm_setzero_pd(), imm8, k, a, b, c);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, c, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fixupimm_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_fixupimm_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_fixupimm_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_fixupimm_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_fixupimm_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_fixupimm_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_fixupimm_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_fixupimm_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_fixupimm_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fixupimm_ss)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_fixupimm_ss)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_fixupimm_ss)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fixupimm_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_fixupimm_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_fixupimm_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_fixupimm_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_fixupimm_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_fixupimm_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_fixupimm_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_fixupimm_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_fixupimm_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_fixupimm_sd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_fixupimm_sd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_fixupimm_sd)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
