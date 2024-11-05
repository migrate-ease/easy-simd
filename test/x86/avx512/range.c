#define EASYSIMD_TEST_X86_AVX512_INSN range

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/set.h>
#include <easysimd/x86/avx512/setzero.h>
#include <easysimd/x86/avx512/range.h>

static int
test_easysimd_mm_range_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   255.95), EASYSIMD_FLOAT32_C(   262.13), EASYSIMD_FLOAT32_C(   990.40), EASYSIMD_FLOAT32_C(  -502.54) },
      { EASYSIMD_FLOAT32_C(  -837.49), EASYSIMD_FLOAT32_C(   145.07), EASYSIMD_FLOAT32_C(   710.99), EASYSIMD_FLOAT32_C(  -255.92) },
      { EASYSIMD_FLOAT32_C(   837.49), EASYSIMD_FLOAT32_C(   262.13), EASYSIMD_FLOAT32_C(   990.40), EASYSIMD_FLOAT32_C(   502.54) } },
    { { EASYSIMD_FLOAT32_C(  -196.03), EASYSIMD_FLOAT32_C(   640.68), EASYSIMD_FLOAT32_C(  -138.92), EASYSIMD_FLOAT32_C(  -782.11) },
      { EASYSIMD_FLOAT32_C(  -561.08), EASYSIMD_FLOAT32_C(  -912.21), EASYSIMD_FLOAT32_C(   957.45), EASYSIMD_FLOAT32_C(   859.99) },
      { EASYSIMD_FLOAT32_C(  -561.08), EASYSIMD_FLOAT32_C(  -912.21), EASYSIMD_FLOAT32_C(  -138.92), EASYSIMD_FLOAT32_C(  -782.11) } },
    { { EASYSIMD_FLOAT32_C(   594.70), EASYSIMD_FLOAT32_C(   371.33), EASYSIMD_FLOAT32_C(   362.58), EASYSIMD_FLOAT32_C(  -743.00) },
      { EASYSIMD_FLOAT32_C(  -691.49), EASYSIMD_FLOAT32_C(  -684.68), EASYSIMD_FLOAT32_C(   514.63), EASYSIMD_FLOAT32_C(   797.88) },
      { EASYSIMD_FLOAT32_C(   594.70), EASYSIMD_FLOAT32_C(   371.33), EASYSIMD_FLOAT32_C(   514.63), EASYSIMD_FLOAT32_C(  -797.88) } },
    { { EASYSIMD_FLOAT32_C(  -878.00), EASYSIMD_FLOAT32_C(  -241.00), EASYSIMD_FLOAT32_C(  -713.77), EASYSIMD_FLOAT32_C(   133.71) },
      { EASYSIMD_FLOAT32_C(  -955.11), EASYSIMD_FLOAT32_C(  -342.49), EASYSIMD_FLOAT32_C(   444.74), EASYSIMD_FLOAT32_C(   300.84) },
      { EASYSIMD_FLOAT32_C(  -878.00), EASYSIMD_FLOAT32_C(  -241.00), EASYSIMD_FLOAT32_C(  -444.74), EASYSIMD_FLOAT32_C(   133.71) } },
    { { EASYSIMD_FLOAT32_C(   919.63), EASYSIMD_FLOAT32_C(   435.14), EASYSIMD_FLOAT32_C(   798.30), EASYSIMD_FLOAT32_C(  -917.86) },
      { EASYSIMD_FLOAT32_C(  -419.79), EASYSIMD_FLOAT32_C(   509.28), EASYSIMD_FLOAT32_C(  -173.78), EASYSIMD_FLOAT32_C(   384.18) },
      { EASYSIMD_FLOAT32_C(   919.63), EASYSIMD_FLOAT32_C(   509.28), EASYSIMD_FLOAT32_C(   798.30), EASYSIMD_FLOAT32_C(  -917.86) } },
    { { EASYSIMD_FLOAT32_C(   149.97), EASYSIMD_FLOAT32_C(   687.31), EASYSIMD_FLOAT32_C(   602.07), EASYSIMD_FLOAT32_C(   588.89) },
      { EASYSIMD_FLOAT32_C(   775.09), EASYSIMD_FLOAT32_C(   559.52), EASYSIMD_FLOAT32_C(   448.88), EASYSIMD_FLOAT32_C(   369.80) },
      { EASYSIMD_FLOAT32_C(   775.09), EASYSIMD_FLOAT32_C(   687.31), EASYSIMD_FLOAT32_C(   602.07), EASYSIMD_FLOAT32_C(   588.89) } },
    { { EASYSIMD_FLOAT32_C(   -69.15), EASYSIMD_FLOAT32_C(  -188.54), EASYSIMD_FLOAT32_C(   626.80), EASYSIMD_FLOAT32_C(   239.36) },
      { EASYSIMD_FLOAT32_C(   126.78), EASYSIMD_FLOAT32_C(   141.43), EASYSIMD_FLOAT32_C(    37.24), EASYSIMD_FLOAT32_C(   248.78) },
      { EASYSIMD_FLOAT32_C(   126.78), EASYSIMD_FLOAT32_C(   141.43), EASYSIMD_FLOAT32_C(   626.80), EASYSIMD_FLOAT32_C(   248.78) } },
    { { EASYSIMD_FLOAT32_C(   900.43), EASYSIMD_FLOAT32_C(   323.47), EASYSIMD_FLOAT32_C(  -617.51), EASYSIMD_FLOAT32_C(   945.32) },
      { EASYSIMD_FLOAT32_C(   980.98), EASYSIMD_FLOAT32_C(   827.24), EASYSIMD_FLOAT32_C(   246.16), EASYSIMD_FLOAT32_C(   900.61) },
      { EASYSIMD_FLOAT32_C(   900.43), EASYSIMD_FLOAT32_C(   323.47), EASYSIMD_FLOAT32_C(  -617.51), EASYSIMD_FLOAT32_C(   900.61) } }
  };

  easysimd__m128 a, b, r;

  a = easysimd_mm_loadu_ps(test_vec[0].a);
  b = easysimd_mm_loadu_ps(test_vec[0].b);
  r = easysimd_mm_range_ps(a, b, 11);
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[1].a);
  b = easysimd_mm_loadu_ps(test_vec[1].b);
  r = easysimd_mm_range_ps(a, b, 4);
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[2].a);
  b = easysimd_mm_loadu_ps(test_vec[2].b);
  r = easysimd_mm_range_ps(a, b, 1);
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[3].a);
  b = easysimd_mm_loadu_ps(test_vec[3].b);
  r = easysimd_mm_range_ps(a, b, 2);
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[4].a);
  b = easysimd_mm_loadu_ps(test_vec[4].b);
  r = easysimd_mm_range_ps(a, b, 3);
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[5].a);
  b = easysimd_mm_loadu_ps(test_vec[5].b);
  r = easysimd_mm_range_ps(a, b, 9);
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[6].a);
  b = easysimd_mm_loadu_ps(test_vec[6].b);
  r = easysimd_mm_range_ps(a, b, 5);
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[7].a);
  b = easysimd_mm_loadu_ps(test_vec[7].b);
  r = easysimd_mm_range_ps(a, b, 0);
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);

  easysimd__m128 a, b, r;

  a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm_range_ps(a, b, 11);

  easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm_range_ps(a, b, 4);

  easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm_range_ps(a, b, 1);

  easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm_range_ps(a, b, 2);

  easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm_range_ps(a, b, 3);

  easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm_range_ps(a, b, 9);

  easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm_range_ps(a, b, 5);

  easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm_range_ps(a, b, 0);

  easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  return 1;
#endif
}

static int
test_easysimd_mm_mask_range_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  easysimd__m128 src, a, b, e, r;

  src = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -475.13), EASYSIMD_FLOAT32_C(  -420.22), EASYSIMD_FLOAT32_C(  -562.17), EASYSIMD_FLOAT32_C(   187.68));
  a = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -756.17), EASYSIMD_FLOAT32_C(  -745.80), EASYSIMD_FLOAT32_C(  -452.78), EASYSIMD_FLOAT32_C(   330.45));
  b = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -467.50), EASYSIMD_FLOAT32_C(   434.14), EASYSIMD_FLOAT32_C(   -54.78), EASYSIMD_FLOAT32_C(  -810.43));
  e = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -475.13), EASYSIMD_FLOAT32_C(  -420.22), EASYSIMD_FLOAT32_C(   452.78), EASYSIMD_FLOAT32_C(   187.68));
  r = easysimd_mm_mask_range_ps(src, UINT8_C(194), a, b, INT32_C(          11));
  easysimd_test_x86_assert_equal_f32x4(r, e, 1);

  src = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   437.68), EASYSIMD_FLOAT32_C(   650.60), EASYSIMD_FLOAT32_C(  -352.96), EASYSIMD_FLOAT32_C(   637.19));
  a = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   684.24), EASYSIMD_FLOAT32_C(   201.31), EASYSIMD_FLOAT32_C(  -376.46), EASYSIMD_FLOAT32_C(   518.68));
  b = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   410.87), EASYSIMD_FLOAT32_C(  -185.65), EASYSIMD_FLOAT32_C(  -832.96), EASYSIMD_FLOAT32_C(  -931.61));
  e = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   684.24), EASYSIMD_FLOAT32_C(   201.31), EASYSIMD_FLOAT32_C(  -352.96), EASYSIMD_FLOAT32_C(   518.68));
  r = easysimd_mm_mask_range_ps(src, UINT8_C(205), a, b, INT32_C(           9));
  easysimd_test_x86_assert_equal_f32x4(r, e, 1);

  src = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -981.24), EASYSIMD_FLOAT32_C(   216.99), EASYSIMD_FLOAT32_C(   393.09), EASYSIMD_FLOAT32_C(  -168.92));
  a = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   127.22), EASYSIMD_FLOAT32_C(   555.16), EASYSIMD_FLOAT32_C(  -456.37), EASYSIMD_FLOAT32_C(   796.77));
  b = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   280.43), EASYSIMD_FLOAT32_C(   371.05), EASYSIMD_FLOAT32_C(   809.37), EASYSIMD_FLOAT32_C(    90.86));
  e = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -280.43), EASYSIMD_FLOAT32_C(   216.99), EASYSIMD_FLOAT32_C(   393.09), EASYSIMD_FLOAT32_C(  -796.77));
  r = easysimd_mm_mask_range_ps(src, UINT8_C( 41), a, b, INT32_C(          15));
  easysimd_test_x86_assert_equal_f32x4(r, e, 1);

  src = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -557.61), EASYSIMD_FLOAT32_C(  -637.19), EASYSIMD_FLOAT32_C(   812.93), EASYSIMD_FLOAT32_C(  -194.80));
  a = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   532.09), EASYSIMD_FLOAT32_C(    94.31), EASYSIMD_FLOAT32_C(   880.08), EASYSIMD_FLOAT32_C(  -986.59));
  b = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -427.99), EASYSIMD_FLOAT32_C(   216.33), EASYSIMD_FLOAT32_C(  -704.37), EASYSIMD_FLOAT32_C(  -496.38));
  e = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -557.61), EASYSIMD_FLOAT32_C(   -94.31), EASYSIMD_FLOAT32_C(  -704.37), EASYSIMD_FLOAT32_C(  -986.59));
  r = easysimd_mm_mask_range_ps(src, UINT8_C(119), a, b, INT32_C(          12));
  easysimd_test_x86_assert_equal_f32x4(r, e, 1);

  src = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -138.24), EASYSIMD_FLOAT32_C(   -30.06), EASYSIMD_FLOAT32_C(   982.88), EASYSIMD_FLOAT32_C(  -969.32));
  a = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   983.71), EASYSIMD_FLOAT32_C(   206.90), EASYSIMD_FLOAT32_C(  -119.48), EASYSIMD_FLOAT32_C(  -813.07));
  b = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -484.99), EASYSIMD_FLOAT32_C(   110.93), EASYSIMD_FLOAT32_C(  -237.94), EASYSIMD_FLOAT32_C(   424.15));
  e = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -484.99), EASYSIMD_FLOAT32_C(   -30.06), EASYSIMD_FLOAT32_C(  -119.48), EASYSIMD_FLOAT32_C(  -969.32));
  r = easysimd_mm_mask_range_ps(src, UINT8_C( 58), a, b, INT32_C(           6));
  easysimd_test_x86_assert_equal_f32x4(r, e, 1);

  src = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   287.18), EASYSIMD_FLOAT32_C(   326.02), EASYSIMD_FLOAT32_C(   795.44), EASYSIMD_FLOAT32_C(  -518.02));
  a = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   702.24), EASYSIMD_FLOAT32_C(  -931.66), EASYSIMD_FLOAT32_C(   729.57), EASYSIMD_FLOAT32_C(   688.83));
  b = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -886.73), EASYSIMD_FLOAT32_C(   234.33), EASYSIMD_FLOAT32_C(   162.66), EASYSIMD_FLOAT32_C(   609.65));
  e = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -886.73), EASYSIMD_FLOAT32_C(   326.02), EASYSIMD_FLOAT32_C(   795.44), EASYSIMD_FLOAT32_C(   688.83));
  r = easysimd_mm_mask_range_ps(src, UINT8_C(153), a, b, INT32_C(           7));
  easysimd_test_x86_assert_equal_f32x4(r, e, 1);

  src = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -518.66), EASYSIMD_FLOAT32_C(   920.95), EASYSIMD_FLOAT32_C(  -314.72), EASYSIMD_FLOAT32_C(  -549.34));
  a = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    77.83), EASYSIMD_FLOAT32_C(  -955.87), EASYSIMD_FLOAT32_C(   343.10), EASYSIMD_FLOAT32_C(  -109.11));
  b = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   647.77), EASYSIMD_FLOAT32_C(    61.53), EASYSIMD_FLOAT32_C(   251.02), EASYSIMD_FLOAT32_C(  -776.38));
  e = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -518.66), EASYSIMD_FLOAT32_C(   920.95), EASYSIMD_FLOAT32_C(   251.02), EASYSIMD_FLOAT32_C(  -109.11));
  r = easysimd_mm_mask_range_ps(src, UINT8_C( 99), a, b, INT32_C(           2));
  easysimd_test_x86_assert_equal_f32x4(r, e, 1);

  src = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -345.55), EASYSIMD_FLOAT32_C(  -415.49), EASYSIMD_FLOAT32_C(  -837.22), EASYSIMD_FLOAT32_C(  -827.54));
  a = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   599.36), EASYSIMD_FLOAT32_C(   566.58), EASYSIMD_FLOAT32_C(   941.63), EASYSIMD_FLOAT32_C(   910.53));
  b = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   280.85), EASYSIMD_FLOAT32_C(   301.59), EASYSIMD_FLOAT32_C(   634.93), EASYSIMD_FLOAT32_C(   671.20));
  e = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   280.85), EASYSIMD_FLOAT32_C(  -415.49), EASYSIMD_FLOAT32_C(  -837.22), EASYSIMD_FLOAT32_C(   671.20));
  r = easysimd_mm_mask_range_ps(src, UINT8_C(153), a, b, INT32_C(           8));
  easysimd_test_x86_assert_equal_f32x4(r, e, 1);

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    int imm8 = easysimd_test_codegen_rand() & 15;
    easysimd__m128 r;
    EASYSIMD_CONSTIFY_16_(easysimd_mm_mask_range_ps, r, easysimd_mm_setzero_ps(), imm8, src, k, a, b);

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
test_easysimd_mm_maskz_range_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  easysimd__m128 a, b, e, r;

  a = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -920.60), EASYSIMD_FLOAT32_C(   -13.42), EASYSIMD_FLOAT32_C(  -744.13), EASYSIMD_FLOAT32_C(   394.12));
  b = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    67.71), EASYSIMD_FLOAT32_C(  -252.44), EASYSIMD_FLOAT32_C(   467.92), EASYSIMD_FLOAT32_C(  -823.18));
  e = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -13.42), EASYSIMD_FLOAT32_C(  -467.92), EASYSIMD_FLOAT32_C(     0.00));
  r = easysimd_mm_maskz_range_ps(UINT8_C(134), a, b, INT32_C(           2));
  easysimd_test_x86_assert_equal_f32x4(r, e, 1);

  a = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   207.07), EASYSIMD_FLOAT32_C(  -957.29), EASYSIMD_FLOAT32_C(    34.64), EASYSIMD_FLOAT32_C(  -854.46));
  b = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -154.81), EASYSIMD_FLOAT32_C(   379.53), EASYSIMD_FLOAT32_C(  -944.21), EASYSIMD_FLOAT32_C(  -317.59));
  e = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -154.81), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    34.64), EASYSIMD_FLOAT32_C(     0.00));
  r = easysimd_mm_maskz_range_ps(UINT8_C(170), a, b, INT32_C(           6));
  easysimd_test_x86_assert_equal_f32x4(r, e, 1);

  a = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -630.01), EASYSIMD_FLOAT32_C(   975.61), EASYSIMD_FLOAT32_C(  -449.17), EASYSIMD_FLOAT32_C(  -196.59));
  b = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   451.78), EASYSIMD_FLOAT32_C(  -995.08), EASYSIMD_FLOAT32_C(   646.81), EASYSIMD_FLOAT32_C(  -849.81));
  e = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -975.61), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00));
  r = easysimd_mm_maskz_range_ps(UINT8_C(228), a, b, INT32_C(          13));
  easysimd_test_x86_assert_equal_f32x4(r, e, 1);

  a = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -25.72), EASYSIMD_FLOAT32_C(    58.38), EASYSIMD_FLOAT32_C(  -678.22), EASYSIMD_FLOAT32_C(   987.70));
  b = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   148.74), EASYSIMD_FLOAT32_C(  -557.80), EASYSIMD_FLOAT32_C(   235.20), EASYSIMD_FLOAT32_C(  -598.82));
  e = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   148.74), EASYSIMD_FLOAT32_C(   557.80), EASYSIMD_FLOAT32_C(   678.22), EASYSIMD_FLOAT32_C(     0.00));
  r = easysimd_mm_maskz_range_ps(UINT8_C(206), a, b, INT32_C(          11));
  easysimd_test_x86_assert_equal_f32x4(r, e, 1);

  a = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   983.13), EASYSIMD_FLOAT32_C(  -712.14), EASYSIMD_FLOAT32_C(  -551.55), EASYSIMD_FLOAT32_C(   940.42));
  b = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    35.05), EASYSIMD_FLOAT32_C(  -961.08), EASYSIMD_FLOAT32_C(   -29.73), EASYSIMD_FLOAT32_C(   655.52));
  e = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   -35.05), EASYSIMD_FLOAT32_C(  -712.14), EASYSIMD_FLOAT32_C(   -29.73), EASYSIMD_FLOAT32_C(  -655.52));
  r = easysimd_mm_maskz_range_ps(UINT8_C(111), a, b, INT32_C(          14));
  easysimd_test_x86_assert_equal_f32x4(r, e, 1);

  a = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    44.64), EASYSIMD_FLOAT32_C(   230.06), EASYSIMD_FLOAT32_C(  -381.13), EASYSIMD_FLOAT32_C(    69.03));
  b = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(    -6.22), EASYSIMD_FLOAT32_C(  -308.55), EASYSIMD_FLOAT32_C(   380.25), EASYSIMD_FLOAT32_C(   -11.14));
  e = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -381.13), EASYSIMD_FLOAT32_C(    69.03));
  r = easysimd_mm_maskz_range_ps(UINT8_C( 51), a, b, INT32_C(           7));
  easysimd_test_x86_assert_equal_f32x4(r, e, 1);

  a = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -145.33), EASYSIMD_FLOAT32_C(   940.89), EASYSIMD_FLOAT32_C(  -180.26), EASYSIMD_FLOAT32_C(   796.29));
  b = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -763.78), EASYSIMD_FLOAT32_C(  -910.13), EASYSIMD_FLOAT32_C(  -657.93), EASYSIMD_FLOAT32_C(   794.02));
  e = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(  -145.33), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -180.26), EASYSIMD_FLOAT32_C(   794.02));
  r = easysimd_mm_maskz_range_ps(UINT8_C( 75), a, b, INT32_C(           6));
  easysimd_test_x86_assert_equal_f32x4(r, e, 1);

  a = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   777.30), EASYSIMD_FLOAT32_C(  -158.78), EASYSIMD_FLOAT32_C(   431.22), EASYSIMD_FLOAT32_C(   489.44));
  b = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   453.27), EASYSIMD_FLOAT32_C(  -252.42), EASYSIMD_FLOAT32_C(  -503.26), EASYSIMD_FLOAT32_C(   414.35));
  e = easysimd_mm_set_ps(EASYSIMD_FLOAT32_C(   453.27), EASYSIMD_FLOAT32_C(   158.78), EASYSIMD_FLOAT32_C(   431.22), EASYSIMD_FLOAT32_C(   414.35));
  r = easysimd_mm_maskz_range_ps(UINT8_C( 47), a, b, INT32_C(          10));
  easysimd_test_x86_assert_equal_f32x4(r, e, 1);

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    int imm8 = easysimd_test_codegen_rand() & 15;
    easysimd__m128 r;
    EASYSIMD_CONSTIFY_16_(easysimd_mm_maskz_range_ps, r, easysimd_mm_setzero_ps(), imm8, k, a, b);

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
test_easysimd_mm256_range_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[8];
    const easysimd_float32 b[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   965.90), EASYSIMD_FLOAT32_C(   639.20), EASYSIMD_FLOAT32_C(   459.04), EASYSIMD_FLOAT32_C(  -520.02),
        EASYSIMD_FLOAT32_C(  -921.38), EASYSIMD_FLOAT32_C(  -256.36), EASYSIMD_FLOAT32_C(  -322.58), EASYSIMD_FLOAT32_C(  -975.13) },
      { EASYSIMD_FLOAT32_C(  -835.96), EASYSIMD_FLOAT32_C(   794.37), EASYSIMD_FLOAT32_C(   868.62), EASYSIMD_FLOAT32_C(  -546.74),
        EASYSIMD_FLOAT32_C(  -578.43), EASYSIMD_FLOAT32_C(   511.48), EASYSIMD_FLOAT32_C(  -160.60), EASYSIMD_FLOAT32_C(   388.19) },
      { EASYSIMD_FLOAT32_C(   965.90), EASYSIMD_FLOAT32_C(   794.37), EASYSIMD_FLOAT32_C(   868.62), EASYSIMD_FLOAT32_C(   546.74),
        EASYSIMD_FLOAT32_C(   921.38), EASYSIMD_FLOAT32_C(   511.48), EASYSIMD_FLOAT32_C(   322.58), EASYSIMD_FLOAT32_C(   975.13) } },
    { { EASYSIMD_FLOAT32_C(   419.06), EASYSIMD_FLOAT32_C(   212.58), EASYSIMD_FLOAT32_C(  -437.40), EASYSIMD_FLOAT32_C(  -767.85),
        EASYSIMD_FLOAT32_C(   542.50), EASYSIMD_FLOAT32_C(   326.58), EASYSIMD_FLOAT32_C(  -844.98), EASYSIMD_FLOAT32_C(  -122.00) },
      { EASYSIMD_FLOAT32_C(   668.38), EASYSIMD_FLOAT32_C(   663.21), EASYSIMD_FLOAT32_C(   157.32), EASYSIMD_FLOAT32_C(  -776.07),
        EASYSIMD_FLOAT32_C(   123.57), EASYSIMD_FLOAT32_C(   356.87), EASYSIMD_FLOAT32_C(   896.03), EASYSIMD_FLOAT32_C(    89.48) },
      { EASYSIMD_FLOAT32_C(   419.06), EASYSIMD_FLOAT32_C(   212.58), EASYSIMD_FLOAT32_C(  -437.40), EASYSIMD_FLOAT32_C(  -776.07),
        EASYSIMD_FLOAT32_C(   123.57), EASYSIMD_FLOAT32_C(   326.58), EASYSIMD_FLOAT32_C(  -844.98), EASYSIMD_FLOAT32_C(  -122.00) } },
    { { EASYSIMD_FLOAT32_C(    -3.93), EASYSIMD_FLOAT32_C(   355.07), EASYSIMD_FLOAT32_C(   569.46), EASYSIMD_FLOAT32_C(    74.69),
        EASYSIMD_FLOAT32_C(  -901.29), EASYSIMD_FLOAT32_C(  -753.12), EASYSIMD_FLOAT32_C(    99.55), EASYSIMD_FLOAT32_C(  -737.26) },
      { EASYSIMD_FLOAT32_C(  -958.76), EASYSIMD_FLOAT32_C(   -31.83), EASYSIMD_FLOAT32_C(  -284.00), EASYSIMD_FLOAT32_C(  -537.19),
        EASYSIMD_FLOAT32_C(  -520.35), EASYSIMD_FLOAT32_C(   555.40), EASYSIMD_FLOAT32_C(   851.00), EASYSIMD_FLOAT32_C(   898.71) },
      { EASYSIMD_FLOAT32_C(    -3.93), EASYSIMD_FLOAT32_C(   355.07), EASYSIMD_FLOAT32_C(   569.46), EASYSIMD_FLOAT32_C(    74.69),
        EASYSIMD_FLOAT32_C(  -520.35), EASYSIMD_FLOAT32_C(  -555.40), EASYSIMD_FLOAT32_C(   851.00), EASYSIMD_FLOAT32_C(  -898.71) } },
    { { EASYSIMD_FLOAT32_C(  -232.02), EASYSIMD_FLOAT32_C(  -586.40), EASYSIMD_FLOAT32_C(  -869.13), EASYSIMD_FLOAT32_C(  -689.52),
        EASYSIMD_FLOAT32_C(   740.18), EASYSIMD_FLOAT32_C(  -714.11), EASYSIMD_FLOAT32_C(   188.48), EASYSIMD_FLOAT32_C(   408.56) },
      { EASYSIMD_FLOAT32_C(   949.10), EASYSIMD_FLOAT32_C(  -654.19), EASYSIMD_FLOAT32_C(   632.49), EASYSIMD_FLOAT32_C(    72.67),
        EASYSIMD_FLOAT32_C(   702.67), EASYSIMD_FLOAT32_C(   528.51), EASYSIMD_FLOAT32_C(  -837.85), EASYSIMD_FLOAT32_C(  -301.26) },
      { EASYSIMD_FLOAT32_C(  -232.02), EASYSIMD_FLOAT32_C(  -586.40), EASYSIMD_FLOAT32_C(  -632.49), EASYSIMD_FLOAT32_C(   -72.67),
        EASYSIMD_FLOAT32_C(   702.67), EASYSIMD_FLOAT32_C(  -528.51), EASYSIMD_FLOAT32_C(   188.48), EASYSIMD_FLOAT32_C(   301.26) } },
    { { EASYSIMD_FLOAT32_C(  -116.42), EASYSIMD_FLOAT32_C(   731.60), EASYSIMD_FLOAT32_C(   773.42), EASYSIMD_FLOAT32_C(   -17.71),
        EASYSIMD_FLOAT32_C(   978.48), EASYSIMD_FLOAT32_C(  -127.03), EASYSIMD_FLOAT32_C(   245.03), EASYSIMD_FLOAT32_C(  -980.28) },
      { EASYSIMD_FLOAT32_C(   841.14), EASYSIMD_FLOAT32_C(   961.03), EASYSIMD_FLOAT32_C(  -517.47), EASYSIMD_FLOAT32_C(  -679.21),
        EASYSIMD_FLOAT32_C(   516.43), EASYSIMD_FLOAT32_C(  -666.47), EASYSIMD_FLOAT32_C(  -780.50), EASYSIMD_FLOAT32_C(  -715.59) },
      { EASYSIMD_FLOAT32_C(  -841.14), EASYSIMD_FLOAT32_C(   961.03), EASYSIMD_FLOAT32_C(   773.42), EASYSIMD_FLOAT32_C(  -679.21),
        EASYSIMD_FLOAT32_C(   978.48), EASYSIMD_FLOAT32_C(  -666.47), EASYSIMD_FLOAT32_C(   780.50), EASYSIMD_FLOAT32_C(  -980.28) } },
    { { EASYSIMD_FLOAT32_C(  -252.87), EASYSIMD_FLOAT32_C(  -649.63), EASYSIMD_FLOAT32_C(  -405.12), EASYSIMD_FLOAT32_C(  -512.69),
        EASYSIMD_FLOAT32_C(  -363.74), EASYSIMD_FLOAT32_C(   783.36), EASYSIMD_FLOAT32_C(   895.86), EASYSIMD_FLOAT32_C(  -414.64) },
      { EASYSIMD_FLOAT32_C(  -870.83), EASYSIMD_FLOAT32_C(   528.35), EASYSIMD_FLOAT32_C(   658.03), EASYSIMD_FLOAT32_C(   831.84),
        EASYSIMD_FLOAT32_C(    56.86), EASYSIMD_FLOAT32_C(   820.17), EASYSIMD_FLOAT32_C(  -469.42), EASYSIMD_FLOAT32_C(   940.44) },
      { EASYSIMD_FLOAT32_C(   252.87), EASYSIMD_FLOAT32_C(   528.35), EASYSIMD_FLOAT32_C(   658.03), EASYSIMD_FLOAT32_C(   831.84),
        EASYSIMD_FLOAT32_C(    56.86), EASYSIMD_FLOAT32_C(   820.17), EASYSIMD_FLOAT32_C(   895.86), EASYSIMD_FLOAT32_C(   940.44) } },
    { { EASYSIMD_FLOAT32_C(   551.78), EASYSIMD_FLOAT32_C(  -696.00), EASYSIMD_FLOAT32_C(   -77.27), EASYSIMD_FLOAT32_C(   530.26),
        EASYSIMD_FLOAT32_C(   176.97), EASYSIMD_FLOAT32_C(  -832.24), EASYSIMD_FLOAT32_C(   549.98), EASYSIMD_FLOAT32_C(    18.12) },
      { EASYSIMD_FLOAT32_C(  -871.21), EASYSIMD_FLOAT32_C(  -967.48), EASYSIMD_FLOAT32_C(   338.91), EASYSIMD_FLOAT32_C(   645.21),
        EASYSIMD_FLOAT32_C(  -633.95), EASYSIMD_FLOAT32_C(   558.41), EASYSIMD_FLOAT32_C(   929.62), EASYSIMD_FLOAT32_C(   113.18) },
      { EASYSIMD_FLOAT32_C(   551.78), EASYSIMD_FLOAT32_C(  -696.00), EASYSIMD_FLOAT32_C(   338.91), EASYSIMD_FLOAT32_C(   645.21),
        EASYSIMD_FLOAT32_C(   176.97), EASYSIMD_FLOAT32_C(   558.41), EASYSIMD_FLOAT32_C(   929.62), EASYSIMD_FLOAT32_C(   113.18) } },
    { { EASYSIMD_FLOAT32_C(   908.78), EASYSIMD_FLOAT32_C(  -475.50), EASYSIMD_FLOAT32_C(   600.49), EASYSIMD_FLOAT32_C(  -454.96),
        EASYSIMD_FLOAT32_C(  -692.13), EASYSIMD_FLOAT32_C(   496.35), EASYSIMD_FLOAT32_C(   130.40), EASYSIMD_FLOAT32_C(  -562.96) },
      { EASYSIMD_FLOAT32_C(    24.70), EASYSIMD_FLOAT32_C(  -211.58), EASYSIMD_FLOAT32_C(  -731.12), EASYSIMD_FLOAT32_C(  -918.44),
        EASYSIMD_FLOAT32_C(  -391.40), EASYSIMD_FLOAT32_C(  -200.55), EASYSIMD_FLOAT32_C(  -978.00), EASYSIMD_FLOAT32_C(  -839.63) },
      { EASYSIMD_FLOAT32_C(    24.70), EASYSIMD_FLOAT32_C(  -475.50), EASYSIMD_FLOAT32_C(   731.12), EASYSIMD_FLOAT32_C(  -918.44),
        EASYSIMD_FLOAT32_C(  -692.13), EASYSIMD_FLOAT32_C(   200.55), EASYSIMD_FLOAT32_C(   978.00), EASYSIMD_FLOAT32_C(  -839.63) } }
  };

  easysimd__m256 a, b, r;

  a = easysimd_mm256_loadu_ps(test_vec[0].a);
  b = easysimd_mm256_loadu_ps(test_vec[0].b);
  r = easysimd_mm256_range_ps(a, b, 11);
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[1].a);
  b = easysimd_mm256_loadu_ps(test_vec[1].b);
  r = easysimd_mm256_range_ps(a, b, 4);
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[2].a);
  b = easysimd_mm256_loadu_ps(test_vec[2].b);
  r = easysimd_mm256_range_ps(a, b, 1);
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[3].a);
  b = easysimd_mm256_loadu_ps(test_vec[3].b);
  r = easysimd_mm256_range_ps(a, b, 2);
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[4].a);
  b = easysimd_mm256_loadu_ps(test_vec[4].b);
  r = easysimd_mm256_range_ps(a, b, 3);
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[5].a);
  b = easysimd_mm256_loadu_ps(test_vec[5].b);
  r = easysimd_mm256_range_ps(a, b, 9);
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[6].a);
  b = easysimd_mm256_loadu_ps(test_vec[6].b);
  r = easysimd_mm256_range_ps(a, b, 5);
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm256_loadu_ps(test_vec[7].a);
  b = easysimd_mm256_loadu_ps(test_vec[7].b);
  r = easysimd_mm256_range_ps(a, b, 0);
  easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);

  easysimd__m256 a, b, r;

  a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm256_range_ps(a, b, 11);

  easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm256_range_ps(a, b, 4);

  easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm256_range_ps(a, b, 1);

  easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm256_range_ps(a, b, 2);

  easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm256_range_ps(a, b, 3);

  easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm256_range_ps(a, b, 9);

  easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm256_range_ps(a, b, 5);

  easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm256_range_ps(a, b, 0);

  easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  return 1;
#endif
}

static int
test_easysimd_mm256_mask_range_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  easysimd__m256 src, a, b, e, r;

  src = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -736.24), EASYSIMD_FLOAT32_C(    60.00), EASYSIMD_FLOAT32_C(   504.80), EASYSIMD_FLOAT32_C(   362.18),
                           EASYSIMD_FLOAT32_C(  -740.22), EASYSIMD_FLOAT32_C(  -433.62), EASYSIMD_FLOAT32_C(   351.91), EASYSIMD_FLOAT32_C(   446.36));
  a = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -438.91), EASYSIMD_FLOAT32_C(  -937.86), EASYSIMD_FLOAT32_C(  -320.64), EASYSIMD_FLOAT32_C(   183.88),
                         EASYSIMD_FLOAT32_C(   224.60), EASYSIMD_FLOAT32_C(   366.59), EASYSIMD_FLOAT32_C(  -844.51), EASYSIMD_FLOAT32_C(   232.20));
  b = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -351.80), EASYSIMD_FLOAT32_C(  -734.27), EASYSIMD_FLOAT32_C(  -415.40), EASYSIMD_FLOAT32_C(   538.38),
                         EASYSIMD_FLOAT32_C(   402.58), EASYSIMD_FLOAT32_C(   590.16), EASYSIMD_FLOAT32_C(  -748.29), EASYSIMD_FLOAT32_C(  -569.46));
  e = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   351.80), EASYSIMD_FLOAT32_C(   734.27), EASYSIMD_FLOAT32_C(   504.80), EASYSIMD_FLOAT32_C(   362.18),
                         EASYSIMD_FLOAT32_C(   224.60), EASYSIMD_FLOAT32_C(   366.59), EASYSIMD_FLOAT32_C(   351.91), EASYSIMD_FLOAT32_C(   446.36));
  r = easysimd_mm256_mask_range_ps(src, UINT8_C(204), a, b, INT32_C(          10));
  easysimd_test_x86_assert_equal_f32x8(r, e, 1);

  src = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -674.00), EASYSIMD_FLOAT32_C(   723.44), EASYSIMD_FLOAT32_C(  -217.15), EASYSIMD_FLOAT32_C(   759.63),
                           EASYSIMD_FLOAT32_C(  -628.48), EASYSIMD_FLOAT32_C(   336.49), EASYSIMD_FLOAT32_C(  -150.50), EASYSIMD_FLOAT32_C(   249.24));
  a = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -846.71), EASYSIMD_FLOAT32_C(   504.87), EASYSIMD_FLOAT32_C(   334.83), EASYSIMD_FLOAT32_C(  -213.29),
                         EASYSIMD_FLOAT32_C(   349.37), EASYSIMD_FLOAT32_C(  -897.38), EASYSIMD_FLOAT32_C(   830.80), EASYSIMD_FLOAT32_C(    85.61));
  b = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -160.01), EASYSIMD_FLOAT32_C(  -126.73), EASYSIMD_FLOAT32_C(   263.20), EASYSIMD_FLOAT32_C(   249.84),
                         EASYSIMD_FLOAT32_C(  -378.44), EASYSIMD_FLOAT32_C(  -167.35), EASYSIMD_FLOAT32_C(  -311.26), EASYSIMD_FLOAT32_C(  -440.57));
  e = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -674.00), EASYSIMD_FLOAT32_C(   126.73), EASYSIMD_FLOAT32_C(   263.20), EASYSIMD_FLOAT32_C(   759.63),
                         EASYSIMD_FLOAT32_C(   378.44), EASYSIMD_FLOAT32_C(   897.38), EASYSIMD_FLOAT32_C(  -150.50), EASYSIMD_FLOAT32_C(   249.24));
  r = easysimd_mm256_mask_range_ps(src, UINT8_C(108), a, b, INT32_C(           8));
  easysimd_test_x86_assert_equal_f32x8(r, e, 1);

  src = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -346.67), EASYSIMD_FLOAT32_C(   909.34), EASYSIMD_FLOAT32_C(  -819.25), EASYSIMD_FLOAT32_C(   316.84),
                           EASYSIMD_FLOAT32_C(    59.85), EASYSIMD_FLOAT32_C(   -68.49), EASYSIMD_FLOAT32_C(   424.59), EASYSIMD_FLOAT32_C(  -588.35));
  a = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -418.58), EASYSIMD_FLOAT32_C(   825.78), EASYSIMD_FLOAT32_C(   361.32), EASYSIMD_FLOAT32_C(  -521.20),
                         EASYSIMD_FLOAT32_C(   994.98), EASYSIMD_FLOAT32_C(  -724.29), EASYSIMD_FLOAT32_C(   436.17), EASYSIMD_FLOAT32_C(   668.97));
  b = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   598.43), EASYSIMD_FLOAT32_C(   -95.70), EASYSIMD_FLOAT32_C(  -524.33), EASYSIMD_FLOAT32_C(  -234.22),
                         EASYSIMD_FLOAT32_C(  -784.44), EASYSIMD_FLOAT32_C(   916.25), EASYSIMD_FLOAT32_C(  -387.51), EASYSIMD_FLOAT32_C(  -289.31));
  e = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -346.67), EASYSIMD_FLOAT32_C(   909.34), EASYSIMD_FLOAT32_C(  -819.25), EASYSIMD_FLOAT32_C(  -234.22),
                         EASYSIMD_FLOAT32_C(    59.85), EASYSIMD_FLOAT32_C(   916.25), EASYSIMD_FLOAT32_C(   424.59), EASYSIMD_FLOAT32_C(  -588.35));
  r = easysimd_mm256_mask_range_ps(src, UINT8_C( 20), a, b, INT32_C(           5));
  easysimd_test_x86_assert_equal_f32x8(r, e, 1);

  src = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -541.09), EASYSIMD_FLOAT32_C(  -581.28), EASYSIMD_FLOAT32_C(  -617.85), EASYSIMD_FLOAT32_C(   527.40),
                           EASYSIMD_FLOAT32_C(    -5.87), EASYSIMD_FLOAT32_C(   970.51), EASYSIMD_FLOAT32_C(  -138.37), EASYSIMD_FLOAT32_C(  -845.86));
  a = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   467.64), EASYSIMD_FLOAT32_C(   825.06), EASYSIMD_FLOAT32_C(    20.32), EASYSIMD_FLOAT32_C(   191.93),
                         EASYSIMD_FLOAT32_C(  -611.11), EASYSIMD_FLOAT32_C(   351.34), EASYSIMD_FLOAT32_C(  -360.34), EASYSIMD_FLOAT32_C(   735.56));
  b = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -198.47), EASYSIMD_FLOAT32_C(   453.56), EASYSIMD_FLOAT32_C(   539.66), EASYSIMD_FLOAT32_C(  -114.72),
                         EASYSIMD_FLOAT32_C(  -158.93), EASYSIMD_FLOAT32_C(  -171.04), EASYSIMD_FLOAT32_C(  -696.14), EASYSIMD_FLOAT32_C(    15.29));
  e = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -541.09), EASYSIMD_FLOAT32_C(  -581.28), EASYSIMD_FLOAT32_C(  -539.66), EASYSIMD_FLOAT32_C(   527.40),
                         EASYSIMD_FLOAT32_C(  -611.11), EASYSIMD_FLOAT32_C(  -351.34), EASYSIMD_FLOAT32_C(  -696.14), EASYSIMD_FLOAT32_C(  -735.56));
  r = easysimd_mm256_mask_range_ps(src, UINT8_C( 47), a, b, INT32_C(          15));
  easysimd_test_x86_assert_equal_f32x8(r, e, 1);

  src = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -655.05), EASYSIMD_FLOAT32_C(  -320.60), EASYSIMD_FLOAT32_C(  -186.34), EASYSIMD_FLOAT32_C(  -625.56),
                           EASYSIMD_FLOAT32_C(   817.77), EASYSIMD_FLOAT32_C(  -340.48), EASYSIMD_FLOAT32_C(   277.20), EASYSIMD_FLOAT32_C(  -780.66));
  a = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   520.44), EASYSIMD_FLOAT32_C(   305.37), EASYSIMD_FLOAT32_C(   962.07), EASYSIMD_FLOAT32_C(  -830.90),
                         EASYSIMD_FLOAT32_C(  -334.29), EASYSIMD_FLOAT32_C(  -773.49), EASYSIMD_FLOAT32_C(  -272.90), EASYSIMD_FLOAT32_C(  -793.20));
  b = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -206.09), EASYSIMD_FLOAT32_C(  -520.12), EASYSIMD_FLOAT32_C(   556.05), EASYSIMD_FLOAT32_C(   964.95),
                         EASYSIMD_FLOAT32_C(  -823.98), EASYSIMD_FLOAT32_C(  -459.24), EASYSIMD_FLOAT32_C(  -502.69), EASYSIMD_FLOAT32_C(  -649.04));
  e = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -655.05), EASYSIMD_FLOAT32_C(  -320.60), EASYSIMD_FLOAT32_C(   556.05), EASYSIMD_FLOAT32_C(  -830.90),
                         EASYSIMD_FLOAT32_C(   817.77), EASYSIMD_FLOAT32_C(  -340.48), EASYSIMD_FLOAT32_C(  -272.90), EASYSIMD_FLOAT32_C(  -780.66));
  r = easysimd_mm256_mask_range_ps(src, UINT8_C( 50), a, b, INT32_C(           2));
  easysimd_test_x86_assert_equal_f32x8(r, e, 1);

  src = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -251.70), EASYSIMD_FLOAT32_C(   443.89), EASYSIMD_FLOAT32_C(  -929.98), EASYSIMD_FLOAT32_C(  -911.22),
                           EASYSIMD_FLOAT32_C(  -833.31), EASYSIMD_FLOAT32_C(   850.68), EASYSIMD_FLOAT32_C(  -666.44), EASYSIMD_FLOAT32_C(   365.16));
  a = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   596.25), EASYSIMD_FLOAT32_C(  -109.62), EASYSIMD_FLOAT32_C(  -226.00), EASYSIMD_FLOAT32_C(   369.74),
                         EASYSIMD_FLOAT32_C(  -836.72), EASYSIMD_FLOAT32_C(  -432.80), EASYSIMD_FLOAT32_C(   561.95), EASYSIMD_FLOAT32_C(   818.34));
  b = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   120.68), EASYSIMD_FLOAT32_C(   242.40), EASYSIMD_FLOAT32_C(   909.28), EASYSIMD_FLOAT32_C(  -420.08),
                         EASYSIMD_FLOAT32_C(  -254.91), EASYSIMD_FLOAT32_C(   558.32), EASYSIMD_FLOAT32_C(    59.48), EASYSIMD_FLOAT32_C(   439.72));
  e = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   120.68), EASYSIMD_FLOAT32_C(   443.89), EASYSIMD_FLOAT32_C(  -226.00), EASYSIMD_FLOAT32_C(  -911.22),
                         EASYSIMD_FLOAT32_C(  -833.31), EASYSIMD_FLOAT32_C(  -432.80), EASYSIMD_FLOAT32_C(    59.48), EASYSIMD_FLOAT32_C(   439.72));
  r = easysimd_mm256_mask_range_ps(src, UINT8_C(167), a, b, INT32_C(           4));
  easysimd_test_x86_assert_equal_f32x8(r, e, 1);

  src = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   -75.46), EASYSIMD_FLOAT32_C(  -665.19), EASYSIMD_FLOAT32_C(   930.33), EASYSIMD_FLOAT32_C(    73.85),
                           EASYSIMD_FLOAT32_C(  -998.75), EASYSIMD_FLOAT32_C(  -434.83), EASYSIMD_FLOAT32_C(  -323.27), EASYSIMD_FLOAT32_C(   207.34));
  a = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   449.55), EASYSIMD_FLOAT32_C(  -266.16), EASYSIMD_FLOAT32_C(   359.25), EASYSIMD_FLOAT32_C(  -117.65),
                         EASYSIMD_FLOAT32_C(   171.89), EASYSIMD_FLOAT32_C(   540.92), EASYSIMD_FLOAT32_C(    -5.44), EASYSIMD_FLOAT32_C(  -576.41));
  b = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   258.14), EASYSIMD_FLOAT32_C(   472.40), EASYSIMD_FLOAT32_C(   663.27), EASYSIMD_FLOAT32_C(   699.83),
                         EASYSIMD_FLOAT32_C(  -587.08), EASYSIMD_FLOAT32_C(  -776.45), EASYSIMD_FLOAT32_C(  -896.42), EASYSIMD_FLOAT32_C(   522.54));
  e = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   449.55), EASYSIMD_FLOAT32_C(  -665.19), EASYSIMD_FLOAT32_C(   930.33), EASYSIMD_FLOAT32_C(  -699.83),
                         EASYSIMD_FLOAT32_C(  -998.75), EASYSIMD_FLOAT32_C(  -434.83), EASYSIMD_FLOAT32_C(  -323.27), EASYSIMD_FLOAT32_C(  -522.54));
  r = easysimd_mm256_mask_range_ps(src, UINT8_C(145), a, b, INT32_C(           1));
  easysimd_test_x86_assert_equal_f32x8(r, e, 1);

  src = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   817.88), EASYSIMD_FLOAT32_C(   849.74), EASYSIMD_FLOAT32_C(  -141.90), EASYSIMD_FLOAT32_C(   252.71),
                           EASYSIMD_FLOAT32_C(   173.00), EASYSIMD_FLOAT32_C(   650.75), EASYSIMD_FLOAT32_C(   167.42), EASYSIMD_FLOAT32_C(  -947.68));
  a = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   386.16), EASYSIMD_FLOAT32_C(  -157.32), EASYSIMD_FLOAT32_C(   617.76), EASYSIMD_FLOAT32_C(   845.24),
                         EASYSIMD_FLOAT32_C(   848.12), EASYSIMD_FLOAT32_C(   194.16), EASYSIMD_FLOAT32_C(   748.22), EASYSIMD_FLOAT32_C(   -76.41));
  b = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   398.14), EASYSIMD_FLOAT32_C(   627.07), EASYSIMD_FLOAT32_C(  -732.06), EASYSIMD_FLOAT32_C(   174.59),
                         EASYSIMD_FLOAT32_C(   523.49), EASYSIMD_FLOAT32_C(  -254.59), EASYSIMD_FLOAT32_C(   725.04), EASYSIMD_FLOAT32_C(  -210.35));
  e = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   817.88), EASYSIMD_FLOAT32_C(   849.74), EASYSIMD_FLOAT32_C(  -141.90), EASYSIMD_FLOAT32_C(   174.59),
                         EASYSIMD_FLOAT32_C(   523.49), EASYSIMD_FLOAT32_C(   254.59), EASYSIMD_FLOAT32_C(   725.04), EASYSIMD_FLOAT32_C(  -210.35));
  r = easysimd_mm256_mask_range_ps(src, UINT8_C( 31), a, b, INT32_C(           0));
  easysimd_test_x86_assert_equal_f32x8(r, e, 1);

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 src = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    int imm8 = easysimd_test_codegen_rand() & 15;
    easysimd__m256 r;
    EASYSIMD_CONSTIFY_16_(easysimd_mm256_mask_range_ps, r, easysimd_mm256_setzero_ps(), imm8, src, k, a, b);

    easysimd_test_x86_write_f32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_range_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  easysimd__m256 a, b, e, r;

  a = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   378.60), EASYSIMD_FLOAT32_C(   120.52), EASYSIMD_FLOAT32_C(   752.45), EASYSIMD_FLOAT32_C(  -794.41),
                         EASYSIMD_FLOAT32_C(   469.76), EASYSIMD_FLOAT32_C(  -414.96), EASYSIMD_FLOAT32_C(  -846.73), EASYSIMD_FLOAT32_C(    61.40));
  b = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -967.88), EASYSIMD_FLOAT32_C(  -428.74), EASYSIMD_FLOAT32_C(  -848.08), EASYSIMD_FLOAT32_C(  -162.04),
                         EASYSIMD_FLOAT32_C(  -176.95), EASYSIMD_FLOAT32_C(   228.33), EASYSIMD_FLOAT32_C(   978.61), EASYSIMD_FLOAT32_C(     5.16));
  e = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   428.74), EASYSIMD_FLOAT32_C(   848.08), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(   469.76), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00));
  r = easysimd_mm256_maskz_range_ps(UINT8_C(104), a, b, INT32_C(           3));
  easysimd_test_x86_assert_equal_f32x8(r, e, 1);

  a = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   742.35), EASYSIMD_FLOAT32_C(   963.02), EASYSIMD_FLOAT32_C(  -451.94), EASYSIMD_FLOAT32_C(  -432.24),
                         EASYSIMD_FLOAT32_C(  -560.47), EASYSIMD_FLOAT32_C(   802.66), EASYSIMD_FLOAT32_C(  -157.27), EASYSIMD_FLOAT32_C(   649.88));
  b = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   502.01), EASYSIMD_FLOAT32_C(   650.14), EASYSIMD_FLOAT32_C(  -798.11), EASYSIMD_FLOAT32_C(   -83.02),
                         EASYSIMD_FLOAT32_C(   496.87), EASYSIMD_FLOAT32_C(   140.49), EASYSIMD_FLOAT32_C(   590.08), EASYSIMD_FLOAT32_C(  -183.99));
  e = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   742.35), EASYSIMD_FLOAT32_C(   963.02), EASYSIMD_FLOAT32_C(  -451.94), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   590.08), EASYSIMD_FLOAT32_C(     0.00));
  r = easysimd_mm256_maskz_range_ps(UINT8_C(226), a, b, INT32_C(           5));
  easysimd_test_x86_assert_equal_f32x8(r, e, 1);

  a = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   608.75), EASYSIMD_FLOAT32_C(    82.67), EASYSIMD_FLOAT32_C(  -537.34), EASYSIMD_FLOAT32_C(  -229.21),
                         EASYSIMD_FLOAT32_C(  -740.37), EASYSIMD_FLOAT32_C(   234.33), EASYSIMD_FLOAT32_C(  -207.83), EASYSIMD_FLOAT32_C(   254.46));
  b = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   873.09), EASYSIMD_FLOAT32_C(   457.35), EASYSIMD_FLOAT32_C(   290.75), EASYSIMD_FLOAT32_C(  -929.56),
                         EASYSIMD_FLOAT32_C(  -385.38), EASYSIMD_FLOAT32_C(   640.87), EASYSIMD_FLOAT32_C(   653.94), EASYSIMD_FLOAT32_C(  -385.42));
  e = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    82.67), EASYSIMD_FLOAT32_C(   290.75), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   234.33), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   254.46));
  r = easysimd_mm256_maskz_range_ps(UINT8_C(101), a, b, INT32_C(          10));
  easysimd_test_x86_assert_equal_f32x8(r, e, 1);

  a = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -799.64), EASYSIMD_FLOAT32_C(  -265.96), EASYSIMD_FLOAT32_C(   -92.05), EASYSIMD_FLOAT32_C(   283.38),
                         EASYSIMD_FLOAT32_C(   237.17), EASYSIMD_FLOAT32_C(   767.46), EASYSIMD_FLOAT32_C(   693.30), EASYSIMD_FLOAT32_C(  -578.84));
  b = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   474.25), EASYSIMD_FLOAT32_C(   573.67), EASYSIMD_FLOAT32_C(   -43.17), EASYSIMD_FLOAT32_C(  -760.08),
                         EASYSIMD_FLOAT32_C(  -218.50), EASYSIMD_FLOAT32_C(   702.37), EASYSIMD_FLOAT32_C(  -615.82), EASYSIMD_FLOAT32_C(   109.84));
  e = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -799.64), EASYSIMD_FLOAT32_C(  -573.67), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -760.08),
                         EASYSIMD_FLOAT32_C(  -237.17), EASYSIMD_FLOAT32_C(  -767.46), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00));
  r = easysimd_mm256_maskz_range_ps(UINT8_C(220), a, b, INT32_C(          15));
  easysimd_test_x86_assert_equal_f32x8(r, e, 1);

  a = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -976.49), EASYSIMD_FLOAT32_C(   166.11), EASYSIMD_FLOAT32_C(   594.07), EASYSIMD_FLOAT32_C(   953.07),
                         EASYSIMD_FLOAT32_C(  -448.51), EASYSIMD_FLOAT32_C(   953.20), EASYSIMD_FLOAT32_C(  -700.87), EASYSIMD_FLOAT32_C(   936.91));
  b = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -583.96), EASYSIMD_FLOAT32_C(  -691.59), EASYSIMD_FLOAT32_C(  -682.24), EASYSIMD_FLOAT32_C(  -351.42),
                         EASYSIMD_FLOAT32_C(  -384.89), EASYSIMD_FLOAT32_C(   896.61), EASYSIMD_FLOAT32_C(  -376.54), EASYSIMD_FLOAT32_C(  -115.17));
  e = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   953.07),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   953.20), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   936.91));
  r = easysimd_mm256_maskz_range_ps(UINT8_C( 21), a, b, INT32_C(           3));
  easysimd_test_x86_assert_equal_f32x8(r, e, 1);

  a = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   -86.92), EASYSIMD_FLOAT32_C(   215.33), EASYSIMD_FLOAT32_C(   494.51), EASYSIMD_FLOAT32_C(  -326.84),
                         EASYSIMD_FLOAT32_C(  -566.17), EASYSIMD_FLOAT32_C(   792.14), EASYSIMD_FLOAT32_C(  -711.02), EASYSIMD_FLOAT32_C(   323.99));
  b = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(    86.65), EASYSIMD_FLOAT32_C(   966.93), EASYSIMD_FLOAT32_C(  -675.77), EASYSIMD_FLOAT32_C(   133.45),
                         EASYSIMD_FLOAT32_C(   667.80), EASYSIMD_FLOAT32_C(  -612.68), EASYSIMD_FLOAT32_C(  -211.00), EASYSIMD_FLOAT32_C(  -548.66));
  e = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -675.77), EASYSIMD_FLOAT32_C(  -326.84),
                         EASYSIMD_FLOAT32_C(  -566.17), EASYSIMD_FLOAT32_C(  -612.68), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -548.66));
  r = easysimd_mm256_maskz_range_ps(UINT8_C( 61), a, b, INT32_C(           4));
  easysimd_test_x86_assert_equal_f32x8(r, e, 1);

  a = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   313.87), EASYSIMD_FLOAT32_C(  -819.34), EASYSIMD_FLOAT32_C(   840.12), EASYSIMD_FLOAT32_C(  -334.71),
                         EASYSIMD_FLOAT32_C(   565.55), EASYSIMD_FLOAT32_C(   943.51), EASYSIMD_FLOAT32_C(  -958.17), EASYSIMD_FLOAT32_C(  -319.28));
  b = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   872.99), EASYSIMD_FLOAT32_C(  -998.21), EASYSIMD_FLOAT32_C(    53.90), EASYSIMD_FLOAT32_C(  -919.15),
                         EASYSIMD_FLOAT32_C(   712.82), EASYSIMD_FLOAT32_C(   729.91), EASYSIMD_FLOAT32_C(  -510.94), EASYSIMD_FLOAT32_C(  -842.12));
  e = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -872.99), EASYSIMD_FLOAT32_C(  -819.34), EASYSIMD_FLOAT32_C(  -840.12), EASYSIMD_FLOAT32_C(  -334.71),
                         EASYSIMD_FLOAT32_C(  -712.82), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -319.28));
  r = easysimd_mm256_maskz_range_ps(UINT8_C(249), a, b, INT32_C(          13));
  easysimd_test_x86_assert_equal_f32x8(r, e, 1);

  a = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -374.49), EASYSIMD_FLOAT32_C(   486.64), EASYSIMD_FLOAT32_C(   975.36), EASYSIMD_FLOAT32_C(   492.06),
                         EASYSIMD_FLOAT32_C(   818.84), EASYSIMD_FLOAT32_C(   588.03), EASYSIMD_FLOAT32_C(  -296.94), EASYSIMD_FLOAT32_C(   367.50));
  b = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(   317.09), EASYSIMD_FLOAT32_C(   217.13), EASYSIMD_FLOAT32_C(  -607.12), EASYSIMD_FLOAT32_C(   373.58),
                         EASYSIMD_FLOAT32_C(   175.31), EASYSIMD_FLOAT32_C(   712.16), EASYSIMD_FLOAT32_C(   453.57), EASYSIMD_FLOAT32_C(  -700.41));
  e = easysimd_mm256_set_ps(EASYSIMD_FLOAT32_C(  -317.09), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   607.12), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(   175.31), EASYSIMD_FLOAT32_C(   588.03), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   367.50));
  r = easysimd_mm256_maskz_range_ps(UINT8_C(173), a, b, INT32_C(           2));
  easysimd_test_x86_assert_equal_f32x8(r, e, 1);

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 b = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    int imm8 = easysimd_test_codegen_rand() & 15;
    easysimd__m256 r;
    EASYSIMD_CONSTIFY_16_(easysimd_mm256_maskz_range_ps, r, easysimd_mm256_setzero_ps(), imm8, k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_range_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[16];
    const easysimd_float32 b[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   103.45), EASYSIMD_FLOAT32_C(   -55.27), EASYSIMD_FLOAT32_C(   690.63), EASYSIMD_FLOAT32_C(  -719.57),
        EASYSIMD_FLOAT32_C(   112.48), EASYSIMD_FLOAT32_C(   240.61), EASYSIMD_FLOAT32_C(   298.55), EASYSIMD_FLOAT32_C(   241.27),
        EASYSIMD_FLOAT32_C(   273.12), EASYSIMD_FLOAT32_C(  -362.54), EASYSIMD_FLOAT32_C(  -113.52), EASYSIMD_FLOAT32_C(   639.17),
        EASYSIMD_FLOAT32_C(  -804.13), EASYSIMD_FLOAT32_C(  -183.90), EASYSIMD_FLOAT32_C(  -247.65), EASYSIMD_FLOAT32_C(  -895.35) },
      { EASYSIMD_FLOAT32_C(   340.61), EASYSIMD_FLOAT32_C(  -647.16), EASYSIMD_FLOAT32_C(  -350.31), EASYSIMD_FLOAT32_C(   648.47),
        EASYSIMD_FLOAT32_C(   849.19), EASYSIMD_FLOAT32_C(   780.09), EASYSIMD_FLOAT32_C(  -914.49), EASYSIMD_FLOAT32_C(  -126.12),
        EASYSIMD_FLOAT32_C(  -431.49), EASYSIMD_FLOAT32_C(  -645.61), EASYSIMD_FLOAT32_C(   -44.56), EASYSIMD_FLOAT32_C(   177.10),
        EASYSIMD_FLOAT32_C(   153.84), EASYSIMD_FLOAT32_C(   -22.56), EASYSIMD_FLOAT32_C(   337.47), EASYSIMD_FLOAT32_C(  -742.70) },
      { EASYSIMD_FLOAT32_C(   340.61), EASYSIMD_FLOAT32_C(   647.16), EASYSIMD_FLOAT32_C(   690.63), EASYSIMD_FLOAT32_C(   719.57),
        EASYSIMD_FLOAT32_C(   849.19), EASYSIMD_FLOAT32_C(   780.09), EASYSIMD_FLOAT32_C(   914.49), EASYSIMD_FLOAT32_C(   241.27),
        EASYSIMD_FLOAT32_C(   431.49), EASYSIMD_FLOAT32_C(   645.61), EASYSIMD_FLOAT32_C(   113.52), EASYSIMD_FLOAT32_C(   639.17),
        EASYSIMD_FLOAT32_C(   804.13), EASYSIMD_FLOAT32_C(   183.90), EASYSIMD_FLOAT32_C(   337.47), EASYSIMD_FLOAT32_C(   895.35) } },
    { { EASYSIMD_FLOAT32_C(   922.17), EASYSIMD_FLOAT32_C(    28.10), EASYSIMD_FLOAT32_C(  -462.28), EASYSIMD_FLOAT32_C(    34.65),
        EASYSIMD_FLOAT32_C(  -731.29), EASYSIMD_FLOAT32_C(   836.27), EASYSIMD_FLOAT32_C(  -724.08), EASYSIMD_FLOAT32_C(   541.84),
        EASYSIMD_FLOAT32_C(  -526.27), EASYSIMD_FLOAT32_C(   162.40), EASYSIMD_FLOAT32_C(   181.01), EASYSIMD_FLOAT32_C(  -330.41),
        EASYSIMD_FLOAT32_C(   978.50), EASYSIMD_FLOAT32_C(   933.36), EASYSIMD_FLOAT32_C(  -225.75), EASYSIMD_FLOAT32_C(   319.11) },
      { EASYSIMD_FLOAT32_C(  -713.80), EASYSIMD_FLOAT32_C(   423.94), EASYSIMD_FLOAT32_C(   -32.42), EASYSIMD_FLOAT32_C(  -864.61),
        EASYSIMD_FLOAT32_C(   204.02), EASYSIMD_FLOAT32_C(    53.09), EASYSIMD_FLOAT32_C(     9.27), EASYSIMD_FLOAT32_C(   772.53),
        EASYSIMD_FLOAT32_C(   407.48), EASYSIMD_FLOAT32_C(   964.71), EASYSIMD_FLOAT32_C(   -50.37), EASYSIMD_FLOAT32_C(  -438.68),
        EASYSIMD_FLOAT32_C(   -57.85), EASYSIMD_FLOAT32_C(  -712.89), EASYSIMD_FLOAT32_C(  -181.38), EASYSIMD_FLOAT32_C(  -135.68) },
      { EASYSIMD_FLOAT32_C(  -713.80), EASYSIMD_FLOAT32_C(    28.10), EASYSIMD_FLOAT32_C(  -462.28), EASYSIMD_FLOAT32_C(  -864.61),
        EASYSIMD_FLOAT32_C(  -731.29), EASYSIMD_FLOAT32_C(    53.09), EASYSIMD_FLOAT32_C(  -724.08), EASYSIMD_FLOAT32_C(   541.84),
        EASYSIMD_FLOAT32_C(  -526.27), EASYSIMD_FLOAT32_C(   162.40), EASYSIMD_FLOAT32_C(   -50.37), EASYSIMD_FLOAT32_C(  -438.68),
        EASYSIMD_FLOAT32_C(   -57.85), EASYSIMD_FLOAT32_C(  -712.89), EASYSIMD_FLOAT32_C(  -225.75), EASYSIMD_FLOAT32_C(  -135.68) } },
    { { EASYSIMD_FLOAT32_C(   315.21), EASYSIMD_FLOAT32_C(   356.35), EASYSIMD_FLOAT32_C(   898.97), EASYSIMD_FLOAT32_C(   583.92),
        EASYSIMD_FLOAT32_C(   192.62), EASYSIMD_FLOAT32_C(  -825.11), EASYSIMD_FLOAT32_C(   125.76), EASYSIMD_FLOAT32_C(   666.34),
        EASYSIMD_FLOAT32_C(   337.29), EASYSIMD_FLOAT32_C(  -693.23), EASYSIMD_FLOAT32_C(  -664.06), EASYSIMD_FLOAT32_C(   315.79),
        EASYSIMD_FLOAT32_C(  -759.87), EASYSIMD_FLOAT32_C(   110.18), EASYSIMD_FLOAT32_C(  -365.10), EASYSIMD_FLOAT32_C(  -473.67) },
      { EASYSIMD_FLOAT32_C(  -465.88), EASYSIMD_FLOAT32_C(   602.48), EASYSIMD_FLOAT32_C(  -338.28), EASYSIMD_FLOAT32_C(   738.14),
        EASYSIMD_FLOAT32_C(  -344.43), EASYSIMD_FLOAT32_C(   670.99), EASYSIMD_FLOAT32_C(   510.67), EASYSIMD_FLOAT32_C(  -936.95),
        EASYSIMD_FLOAT32_C(   635.70), EASYSIMD_FLOAT32_C(  -539.70), EASYSIMD_FLOAT32_C(  -375.62), EASYSIMD_FLOAT32_C(  -422.14),
        EASYSIMD_FLOAT32_C(  -252.59), EASYSIMD_FLOAT32_C(   443.00), EASYSIMD_FLOAT32_C(   442.18), EASYSIMD_FLOAT32_C(  -937.38) },
      { EASYSIMD_FLOAT32_C(   315.21), EASYSIMD_FLOAT32_C(   602.48), EASYSIMD_FLOAT32_C(   898.97), EASYSIMD_FLOAT32_C(   738.14),
        EASYSIMD_FLOAT32_C(   192.62), EASYSIMD_FLOAT32_C(  -670.99), EASYSIMD_FLOAT32_C(   510.67), EASYSIMD_FLOAT32_C(   666.34),
        EASYSIMD_FLOAT32_C(   635.70), EASYSIMD_FLOAT32_C(  -539.70), EASYSIMD_FLOAT32_C(  -375.62), EASYSIMD_FLOAT32_C(   315.79),
        EASYSIMD_FLOAT32_C(  -252.59), EASYSIMD_FLOAT32_C(   443.00), EASYSIMD_FLOAT32_C(  -442.18), EASYSIMD_FLOAT32_C(  -473.67) } },
    { { EASYSIMD_FLOAT32_C(  -200.66), EASYSIMD_FLOAT32_C(   341.14), EASYSIMD_FLOAT32_C(   646.55), EASYSIMD_FLOAT32_C(   991.96),
        EASYSIMD_FLOAT32_C(   516.03), EASYSIMD_FLOAT32_C(  -227.69), EASYSIMD_FLOAT32_C(   658.30), EASYSIMD_FLOAT32_C(  -146.69),
        EASYSIMD_FLOAT32_C(    79.08), EASYSIMD_FLOAT32_C(   994.24), EASYSIMD_FLOAT32_C(  -830.90), EASYSIMD_FLOAT32_C(   319.21),
        EASYSIMD_FLOAT32_C(   104.42), EASYSIMD_FLOAT32_C(  -196.00), EASYSIMD_FLOAT32_C(   845.55), EASYSIMD_FLOAT32_C(   638.54) },
      { EASYSIMD_FLOAT32_C(  -593.52), EASYSIMD_FLOAT32_C(  -492.73), EASYSIMD_FLOAT32_C(   376.68), EASYSIMD_FLOAT32_C(    62.05),
        EASYSIMD_FLOAT32_C(  -821.74), EASYSIMD_FLOAT32_C(  -112.65), EASYSIMD_FLOAT32_C(   125.10), EASYSIMD_FLOAT32_C(   813.97),
        EASYSIMD_FLOAT32_C(   347.66), EASYSIMD_FLOAT32_C(   749.48), EASYSIMD_FLOAT32_C(  -608.18), EASYSIMD_FLOAT32_C(  -904.93),
        EASYSIMD_FLOAT32_C(   192.48), EASYSIMD_FLOAT32_C(   834.00), EASYSIMD_FLOAT32_C(  -842.31), EASYSIMD_FLOAT32_C(   991.82) },
      { EASYSIMD_FLOAT32_C(  -200.66), EASYSIMD_FLOAT32_C(   341.14), EASYSIMD_FLOAT32_C(   376.68), EASYSIMD_FLOAT32_C(    62.05),
        EASYSIMD_FLOAT32_C(   516.03), EASYSIMD_FLOAT32_C(  -112.65), EASYSIMD_FLOAT32_C(   125.10), EASYSIMD_FLOAT32_C(  -146.69),
        EASYSIMD_FLOAT32_C(    79.08), EASYSIMD_FLOAT32_C(   749.48), EASYSIMD_FLOAT32_C(  -608.18), EASYSIMD_FLOAT32_C(   319.21),
        EASYSIMD_FLOAT32_C(   104.42), EASYSIMD_FLOAT32_C(  -196.00), EASYSIMD_FLOAT32_C(   842.31), EASYSIMD_FLOAT32_C(   638.54) } },
    { { EASYSIMD_FLOAT32_C(   175.14), EASYSIMD_FLOAT32_C(   804.24), EASYSIMD_FLOAT32_C(   983.78), EASYSIMD_FLOAT32_C(  -308.83),
        EASYSIMD_FLOAT32_C(  -423.45), EASYSIMD_FLOAT32_C(   642.08), EASYSIMD_FLOAT32_C(   544.49), EASYSIMD_FLOAT32_C(   655.63),
        EASYSIMD_FLOAT32_C(   636.32), EASYSIMD_FLOAT32_C(   713.59), EASYSIMD_FLOAT32_C(   -25.16), EASYSIMD_FLOAT32_C(  -259.25),
        EASYSIMD_FLOAT32_C(  -482.41), EASYSIMD_FLOAT32_C(  -179.61), EASYSIMD_FLOAT32_C(  -620.71), EASYSIMD_FLOAT32_C(   -75.93) },
      { EASYSIMD_FLOAT32_C(   327.66), EASYSIMD_FLOAT32_C(   755.97), EASYSIMD_FLOAT32_C(   986.12), EASYSIMD_FLOAT32_C(   505.92),
        EASYSIMD_FLOAT32_C(  -356.68), EASYSIMD_FLOAT32_C(   111.23), EASYSIMD_FLOAT32_C(   319.89), EASYSIMD_FLOAT32_C(   990.98),
        EASYSIMD_FLOAT32_C(  -139.29), EASYSIMD_FLOAT32_C(   711.72), EASYSIMD_FLOAT32_C(  -913.95), EASYSIMD_FLOAT32_C(  -946.81),
        EASYSIMD_FLOAT32_C(   545.72), EASYSIMD_FLOAT32_C(  -756.26), EASYSIMD_FLOAT32_C(  -954.99), EASYSIMD_FLOAT32_C(  -279.14) },
      { EASYSIMD_FLOAT32_C(   327.66), EASYSIMD_FLOAT32_C(   804.24), EASYSIMD_FLOAT32_C(   986.12), EASYSIMD_FLOAT32_C(  -505.92),
        EASYSIMD_FLOAT32_C(  -423.45), EASYSIMD_FLOAT32_C(   642.08), EASYSIMD_FLOAT32_C(   544.49), EASYSIMD_FLOAT32_C(   990.98),
        EASYSIMD_FLOAT32_C(   636.32), EASYSIMD_FLOAT32_C(   713.59), EASYSIMD_FLOAT32_C(  -913.95), EASYSIMD_FLOAT32_C(  -946.81),
        EASYSIMD_FLOAT32_C(  -545.72), EASYSIMD_FLOAT32_C(  -756.26), EASYSIMD_FLOAT32_C(  -954.99), EASYSIMD_FLOAT32_C(  -279.14) } },
    { { EASYSIMD_FLOAT32_C(  -952.02), EASYSIMD_FLOAT32_C(  -971.20), EASYSIMD_FLOAT32_C(   412.03), EASYSIMD_FLOAT32_C(  -375.47),
        EASYSIMD_FLOAT32_C(   670.88), EASYSIMD_FLOAT32_C(   -43.48), EASYSIMD_FLOAT32_C(  -719.85), EASYSIMD_FLOAT32_C(   307.20),
        EASYSIMD_FLOAT32_C(  -329.89), EASYSIMD_FLOAT32_C(   255.00), EASYSIMD_FLOAT32_C(  -952.05), EASYSIMD_FLOAT32_C(   187.70),
        EASYSIMD_FLOAT32_C(  -924.61), EASYSIMD_FLOAT32_C(  -572.77), EASYSIMD_FLOAT32_C(  -888.23), EASYSIMD_FLOAT32_C(   403.05) },
      { EASYSIMD_FLOAT32_C(  -816.80), EASYSIMD_FLOAT32_C(  -902.10), EASYSIMD_FLOAT32_C(   -91.02), EASYSIMD_FLOAT32_C(  -173.47),
        EASYSIMD_FLOAT32_C(   209.12), EASYSIMD_FLOAT32_C(  -771.13), EASYSIMD_FLOAT32_C(  -182.49), EASYSIMD_FLOAT32_C(  -930.17),
        EASYSIMD_FLOAT32_C(   940.59), EASYSIMD_FLOAT32_C(   -96.45), EASYSIMD_FLOAT32_C(  -876.98), EASYSIMD_FLOAT32_C(   486.30),
        EASYSIMD_FLOAT32_C(   147.29), EASYSIMD_FLOAT32_C(  -831.96), EASYSIMD_FLOAT32_C(  -792.84), EASYSIMD_FLOAT32_C(   195.27) },
      { EASYSIMD_FLOAT32_C(   816.80), EASYSIMD_FLOAT32_C(   902.10), EASYSIMD_FLOAT32_C(   412.03), EASYSIMD_FLOAT32_C(   173.47),
        EASYSIMD_FLOAT32_C(   670.88), EASYSIMD_FLOAT32_C(    43.48), EASYSIMD_FLOAT32_C(   182.49), EASYSIMD_FLOAT32_C(   307.20),
        EASYSIMD_FLOAT32_C(   940.59), EASYSIMD_FLOAT32_C(   255.00), EASYSIMD_FLOAT32_C(   876.98), EASYSIMD_FLOAT32_C(   486.30),
        EASYSIMD_FLOAT32_C(   147.29), EASYSIMD_FLOAT32_C(   572.77), EASYSIMD_FLOAT32_C(   792.84), EASYSIMD_FLOAT32_C(   403.05) } },
    { { EASYSIMD_FLOAT32_C(  -803.17), EASYSIMD_FLOAT32_C(   619.20), EASYSIMD_FLOAT32_C(   819.80), EASYSIMD_FLOAT32_C(   867.71),
        EASYSIMD_FLOAT32_C(  -424.28), EASYSIMD_FLOAT32_C(  -900.05), EASYSIMD_FLOAT32_C(   174.91), EASYSIMD_FLOAT32_C(   245.83),
        EASYSIMD_FLOAT32_C(   354.95), EASYSIMD_FLOAT32_C(   222.86), EASYSIMD_FLOAT32_C(  -566.47), EASYSIMD_FLOAT32_C(   430.34),
        EASYSIMD_FLOAT32_C(   650.10), EASYSIMD_FLOAT32_C(  -454.70), EASYSIMD_FLOAT32_C(  -166.61), EASYSIMD_FLOAT32_C(   833.30) },
      { EASYSIMD_FLOAT32_C(  -356.80), EASYSIMD_FLOAT32_C(   742.37), EASYSIMD_FLOAT32_C(  -340.17), EASYSIMD_FLOAT32_C(   852.32),
        EASYSIMD_FLOAT32_C(   971.24), EASYSIMD_FLOAT32_C(   477.33), EASYSIMD_FLOAT32_C(   922.16), EASYSIMD_FLOAT32_C(   911.83),
        EASYSIMD_FLOAT32_C(  -619.11), EASYSIMD_FLOAT32_C(  -954.82), EASYSIMD_FLOAT32_C(   398.14), EASYSIMD_FLOAT32_C(   528.18),
        EASYSIMD_FLOAT32_C(  -786.79), EASYSIMD_FLOAT32_C(   605.30), EASYSIMD_FLOAT32_C(  -276.55), EASYSIMD_FLOAT32_C(  -589.95) },
      { EASYSIMD_FLOAT32_C(  -356.80), EASYSIMD_FLOAT32_C(   742.37), EASYSIMD_FLOAT32_C(   819.80), EASYSIMD_FLOAT32_C(   867.71),
        EASYSIMD_FLOAT32_C(   971.24), EASYSIMD_FLOAT32_C(   477.33), EASYSIMD_FLOAT32_C(   922.16), EASYSIMD_FLOAT32_C(   911.83),
        EASYSIMD_FLOAT32_C(   354.95), EASYSIMD_FLOAT32_C(   222.86), EASYSIMD_FLOAT32_C(   398.14), EASYSIMD_FLOAT32_C(   528.18),
        EASYSIMD_FLOAT32_C(   650.10), EASYSIMD_FLOAT32_C(   605.30), EASYSIMD_FLOAT32_C(  -166.61), EASYSIMD_FLOAT32_C(   833.30) } },
    { { EASYSIMD_FLOAT32_C(   224.50), EASYSIMD_FLOAT32_C(  -456.75), EASYSIMD_FLOAT32_C(  -722.24), EASYSIMD_FLOAT32_C(   800.22),
        EASYSIMD_FLOAT32_C(  -356.80), EASYSIMD_FLOAT32_C(   452.67), EASYSIMD_FLOAT32_C(    46.04), EASYSIMD_FLOAT32_C(   998.15),
        EASYSIMD_FLOAT32_C(  -324.47), EASYSIMD_FLOAT32_C(   479.57), EASYSIMD_FLOAT32_C(   428.49), EASYSIMD_FLOAT32_C(  -674.37),
        EASYSIMD_FLOAT32_C(  -975.12), EASYSIMD_FLOAT32_C(  -738.12), EASYSIMD_FLOAT32_C(  -841.07), EASYSIMD_FLOAT32_C(  -331.93) },
      { EASYSIMD_FLOAT32_C(  -995.74), EASYSIMD_FLOAT32_C(  -181.24), EASYSIMD_FLOAT32_C(  -479.60), EASYSIMD_FLOAT32_C(   975.50),
        EASYSIMD_FLOAT32_C(  -703.91), EASYSIMD_FLOAT32_C(  -557.45), EASYSIMD_FLOAT32_C(   887.33), EASYSIMD_FLOAT32_C(  -323.02),
        EASYSIMD_FLOAT32_C(  -512.27), EASYSIMD_FLOAT32_C(   285.47), EASYSIMD_FLOAT32_C(  -794.84), EASYSIMD_FLOAT32_C(  -299.05),
        EASYSIMD_FLOAT32_C(  -109.23), EASYSIMD_FLOAT32_C(   -71.39), EASYSIMD_FLOAT32_C(   110.99), EASYSIMD_FLOAT32_C(  -884.73) },
      { EASYSIMD_FLOAT32_C(   995.74), EASYSIMD_FLOAT32_C(  -456.75), EASYSIMD_FLOAT32_C(  -722.24), EASYSIMD_FLOAT32_C(   800.22),
        EASYSIMD_FLOAT32_C(  -703.91), EASYSIMD_FLOAT32_C(   557.45), EASYSIMD_FLOAT32_C(    46.04), EASYSIMD_FLOAT32_C(   323.02),
        EASYSIMD_FLOAT32_C(  -512.27), EASYSIMD_FLOAT32_C(   285.47), EASYSIMD_FLOAT32_C(   794.84), EASYSIMD_FLOAT32_C(  -674.37),
        EASYSIMD_FLOAT32_C(  -975.12), EASYSIMD_FLOAT32_C(  -738.12), EASYSIMD_FLOAT32_C(  -841.07), EASYSIMD_FLOAT32_C(  -884.73) } }
  };

  easysimd__m512 a, b, r;

  a = easysimd_mm512_loadu_ps(test_vec[0].a);
  b = easysimd_mm512_loadu_ps(test_vec[0].b);
  r = easysimd_mm512_range_ps(a, b, 11);
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[1].a);
  b = easysimd_mm512_loadu_ps(test_vec[1].b);
  r = easysimd_mm512_range_ps(a, b, 4);
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[2].a);
  b = easysimd_mm512_loadu_ps(test_vec[2].b);
  r = easysimd_mm512_range_ps(a, b, 1);
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[3].a);
  b = easysimd_mm512_loadu_ps(test_vec[3].b);
  r = easysimd_mm512_range_ps(a, b, 2);
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[4].a);
  b = easysimd_mm512_loadu_ps(test_vec[4].b);
  r = easysimd_mm512_range_ps(a, b, 3);
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[5].a);
  b = easysimd_mm512_loadu_ps(test_vec[5].b);
  r = easysimd_mm512_range_ps(a, b, 9);
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[6].a);
  b = easysimd_mm512_loadu_ps(test_vec[6].b);
  r = easysimd_mm512_range_ps(a, b, 5);
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm512_loadu_ps(test_vec[7].a);
  b = easysimd_mm512_loadu_ps(test_vec[7].b);
  r = easysimd_mm512_range_ps(a, b, 0);
  easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);

  easysimd__m512 a, b, r;

  a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm512_range_ps(a, b, 11);

  easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm512_range_ps(a, b, 4);

  easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm512_range_ps(a, b, 1);

  easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm512_range_ps(a, b, 2);

  easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm512_range_ps(a, b, 3);

  easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm512_range_ps(a, b, 9);

  easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm512_range_ps(a, b, 5);

  easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
  r = easysimd_mm512_range_ps(a, b, 0);

  easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  return 1;
#endif
}

static int
test_easysimd_mm512_mask_range_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  easysimd__m512 src, a, b, e, r;

  src = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -829.11), EASYSIMD_FLOAT32_C(   949.49), EASYSIMD_FLOAT32_C(   704.66), EASYSIMD_FLOAT32_C(   467.83),
                           EASYSIMD_FLOAT32_C(  -418.01), EASYSIMD_FLOAT32_C(    29.71), EASYSIMD_FLOAT32_C(   980.09), EASYSIMD_FLOAT32_C(  -291.00),
                           EASYSIMD_FLOAT32_C(    27.91), EASYSIMD_FLOAT32_C(   -73.81), EASYSIMD_FLOAT32_C(  -371.85), EASYSIMD_FLOAT32_C(   315.10),
                           EASYSIMD_FLOAT32_C(   196.29), EASYSIMD_FLOAT32_C(  -860.91), EASYSIMD_FLOAT32_C(   157.21), EASYSIMD_FLOAT32_C(   882.42));
  a = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -509.00), EASYSIMD_FLOAT32_C(  -443.58), EASYSIMD_FLOAT32_C(   842.51), EASYSIMD_FLOAT32_C(  -648.08),
                         EASYSIMD_FLOAT32_C(   399.21), EASYSIMD_FLOAT32_C(   960.09), EASYSIMD_FLOAT32_C(  -606.51), EASYSIMD_FLOAT32_C(  -917.89),
                         EASYSIMD_FLOAT32_C(  -257.05), EASYSIMD_FLOAT32_C(  -999.39), EASYSIMD_FLOAT32_C(  -291.46), EASYSIMD_FLOAT32_C(   567.65),
                         EASYSIMD_FLOAT32_C(  -711.55), EASYSIMD_FLOAT32_C(   254.97), EASYSIMD_FLOAT32_C(   268.06), EASYSIMD_FLOAT32_C(   662.95));
  b = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   246.74), EASYSIMD_FLOAT32_C(  -872.05), EASYSIMD_FLOAT32_C(   -73.51), EASYSIMD_FLOAT32_C(   583.79),
                         EASYSIMD_FLOAT32_C(  -640.37), EASYSIMD_FLOAT32_C(   633.79), EASYSIMD_FLOAT32_C(   412.91), EASYSIMD_FLOAT32_C(  -589.86),
                         EASYSIMD_FLOAT32_C(   929.13), EASYSIMD_FLOAT32_C(   945.08), EASYSIMD_FLOAT32_C(   828.15), EASYSIMD_FLOAT32_C(  -100.58),
                         EASYSIMD_FLOAT32_C(   964.99), EASYSIMD_FLOAT32_C(   119.15), EASYSIMD_FLOAT32_C(   871.51), EASYSIMD_FLOAT32_C(    38.79));
  e = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -509.00), EASYSIMD_FLOAT32_C(   949.49), EASYSIMD_FLOAT32_C(   842.51), EASYSIMD_FLOAT32_C(   467.83),
                         EASYSIMD_FLOAT32_C(   640.37), EASYSIMD_FLOAT32_C(   960.09), EASYSIMD_FLOAT32_C(   980.09), EASYSIMD_FLOAT32_C(  -917.89),
                         EASYSIMD_FLOAT32_C(  -929.13), EASYSIMD_FLOAT32_C(   -73.81), EASYSIMD_FLOAT32_C(  -371.85), EASYSIMD_FLOAT32_C(   315.10),
                         EASYSIMD_FLOAT32_C(  -964.99), EASYSIMD_FLOAT32_C(   254.97), EASYSIMD_FLOAT32_C(   871.51), EASYSIMD_FLOAT32_C(   882.42));
  r = easysimd_mm512_mask_range_ps(src, UINT16_C(44430), a, b, INT32_C(           3));
  easysimd_test_x86_assert_equal_f32x16(r, e, 1);

  src = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -999.29), EASYSIMD_FLOAT32_C(   346.54), EASYSIMD_FLOAT32_C(  -227.79), EASYSIMD_FLOAT32_C(  -870.81),
                           EASYSIMD_FLOAT32_C(  -692.26), EASYSIMD_FLOAT32_C(  -718.79), EASYSIMD_FLOAT32_C(   572.78), EASYSIMD_FLOAT32_C(  -534.76),
                           EASYSIMD_FLOAT32_C(   929.29), EASYSIMD_FLOAT32_C(  -826.43), EASYSIMD_FLOAT32_C(  -494.85), EASYSIMD_FLOAT32_C(   535.80),
                           EASYSIMD_FLOAT32_C(  -908.54), EASYSIMD_FLOAT32_C(   762.20), EASYSIMD_FLOAT32_C(   535.19), EASYSIMD_FLOAT32_C(   382.92));
  a = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   346.30), EASYSIMD_FLOAT32_C(    35.23), EASYSIMD_FLOAT32_C(  -999.85), EASYSIMD_FLOAT32_C(   584.10),
                         EASYSIMD_FLOAT32_C(   500.04), EASYSIMD_FLOAT32_C(  -382.77), EASYSIMD_FLOAT32_C(   389.55), EASYSIMD_FLOAT32_C(  -746.70),
                         EASYSIMD_FLOAT32_C(  -510.72), EASYSIMD_FLOAT32_C(  -536.94), EASYSIMD_FLOAT32_C(  -330.49), EASYSIMD_FLOAT32_C(  -870.35),
                         EASYSIMD_FLOAT32_C(  -170.74), EASYSIMD_FLOAT32_C(   256.60), EASYSIMD_FLOAT32_C(   719.51), EASYSIMD_FLOAT32_C(   -99.87));
  b = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -132.01), EASYSIMD_FLOAT32_C(   282.49), EASYSIMD_FLOAT32_C(   445.10), EASYSIMD_FLOAT32_C(   967.86),
                         EASYSIMD_FLOAT32_C(   970.96), EASYSIMD_FLOAT32_C(   553.74), EASYSIMD_FLOAT32_C(   967.15), EASYSIMD_FLOAT32_C(  -375.57),
                         EASYSIMD_FLOAT32_C(  -218.47), EASYSIMD_FLOAT32_C(   837.96), EASYSIMD_FLOAT32_C(  -683.31), EASYSIMD_FLOAT32_C(  -499.68),
                         EASYSIMD_FLOAT32_C(  -734.82), EASYSIMD_FLOAT32_C(   851.45), EASYSIMD_FLOAT32_C(  -428.97), EASYSIMD_FLOAT32_C(  -908.39));
  e = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   132.01), EASYSIMD_FLOAT32_C(    35.23), EASYSIMD_FLOAT32_C(  -227.79), EASYSIMD_FLOAT32_C(  -870.81),
                         EASYSIMD_FLOAT32_C(  -692.26), EASYSIMD_FLOAT32_C(  -718.79), EASYSIMD_FLOAT32_C(   572.78), EASYSIMD_FLOAT32_C(  -375.57),
                         EASYSIMD_FLOAT32_C(  -218.47), EASYSIMD_FLOAT32_C(  -826.43), EASYSIMD_FLOAT32_C(  -494.85), EASYSIMD_FLOAT32_C(  -499.68),
                         EASYSIMD_FLOAT32_C(  -170.74), EASYSIMD_FLOAT32_C(   256.60), EASYSIMD_FLOAT32_C(   428.97), EASYSIMD_FLOAT32_C(   382.92));
  r = easysimd_mm512_mask_range_ps(src, UINT16_C(49566), a, b, INT32_C(           2));
  easysimd_test_x86_assert_equal_f32x16(r, e, 1);

  src = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -431.81), EASYSIMD_FLOAT32_C(  -507.46), EASYSIMD_FLOAT32_C(  -519.75), EASYSIMD_FLOAT32_C(   997.16),
                           EASYSIMD_FLOAT32_C(  -599.07), EASYSIMD_FLOAT32_C(   133.95), EASYSIMD_FLOAT32_C(   -38.07), EASYSIMD_FLOAT32_C(  -599.22),
                           EASYSIMD_FLOAT32_C(   549.85), EASYSIMD_FLOAT32_C(   461.89), EASYSIMD_FLOAT32_C(   783.55), EASYSIMD_FLOAT32_C(  -839.70),
                           EASYSIMD_FLOAT32_C(   208.59), EASYSIMD_FLOAT32_C(   294.27), EASYSIMD_FLOAT32_C(   697.25), EASYSIMD_FLOAT32_C(  -460.91));
  a = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   307.78), EASYSIMD_FLOAT32_C(    95.93), EASYSIMD_FLOAT32_C(    65.35), EASYSIMD_FLOAT32_C(  -986.49),
                         EASYSIMD_FLOAT32_C(   398.68), EASYSIMD_FLOAT32_C(  -473.73), EASYSIMD_FLOAT32_C(  -151.11), EASYSIMD_FLOAT32_C(  -469.31),
                         EASYSIMD_FLOAT32_C(   243.78), EASYSIMD_FLOAT32_C(   403.79), EASYSIMD_FLOAT32_C(  -437.17), EASYSIMD_FLOAT32_C(   272.82),
                         EASYSIMD_FLOAT32_C(   850.05), EASYSIMD_FLOAT32_C(  -404.32), EASYSIMD_FLOAT32_C(  -351.61), EASYSIMD_FLOAT32_C(    68.52));
  b = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -668.36), EASYSIMD_FLOAT32_C(   143.30), EASYSIMD_FLOAT32_C(  -248.01), EASYSIMD_FLOAT32_C(   263.12),
                         EASYSIMD_FLOAT32_C(  -614.42), EASYSIMD_FLOAT32_C(  -579.71), EASYSIMD_FLOAT32_C(  -305.07), EASYSIMD_FLOAT32_C(   893.04),
                         EASYSIMD_FLOAT32_C(   940.04), EASYSIMD_FLOAT32_C(  -302.23), EASYSIMD_FLOAT32_C(   492.10), EASYSIMD_FLOAT32_C(  -193.92),
                         EASYSIMD_FLOAT32_C(   735.84), EASYSIMD_FLOAT32_C(    91.32), EASYSIMD_FLOAT32_C(   256.23), EASYSIMD_FLOAT32_C(  -726.05));
  e = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -431.81), EASYSIMD_FLOAT32_C(    95.93), EASYSIMD_FLOAT32_C(    65.35), EASYSIMD_FLOAT32_C(  -263.12),
                         EASYSIMD_FLOAT32_C(  -599.07), EASYSIMD_FLOAT32_C(   133.95), EASYSIMD_FLOAT32_C(   -38.07), EASYSIMD_FLOAT32_C(  -599.22),
                         EASYSIMD_FLOAT32_C(   243.78), EASYSIMD_FLOAT32_C(   461.89), EASYSIMD_FLOAT32_C(  -437.17), EASYSIMD_FLOAT32_C(   193.92),
                         EASYSIMD_FLOAT32_C(   208.59), EASYSIMD_FLOAT32_C(   -91.32), EASYSIMD_FLOAT32_C(   697.25), EASYSIMD_FLOAT32_C(    68.52));
  r = easysimd_mm512_mask_range_ps(src, UINT16_C(28853), a, b, INT32_C(           2));
  easysimd_test_x86_assert_equal_f32x16(r, e, 1);

  src = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -416.66), EASYSIMD_FLOAT32_C(   782.54), EASYSIMD_FLOAT32_C(   755.66), EASYSIMD_FLOAT32_C(   327.11),
                           EASYSIMD_FLOAT32_C(   508.59), EASYSIMD_FLOAT32_C(  -552.12), EASYSIMD_FLOAT32_C(  -768.81), EASYSIMD_FLOAT32_C(  -556.76),
                           EASYSIMD_FLOAT32_C(  -565.63), EASYSIMD_FLOAT32_C(  -167.49), EASYSIMD_FLOAT32_C(   916.97), EASYSIMD_FLOAT32_C(   585.48),
                           EASYSIMD_FLOAT32_C(  -698.18), EASYSIMD_FLOAT32_C(  -326.81), EASYSIMD_FLOAT32_C(  -818.31), EASYSIMD_FLOAT32_C(   738.99));
  a = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -424.68), EASYSIMD_FLOAT32_C(  -312.49), EASYSIMD_FLOAT32_C(   499.99), EASYSIMD_FLOAT32_C(   902.12),
                         EASYSIMD_FLOAT32_C(  -494.18), EASYSIMD_FLOAT32_C(   761.01), EASYSIMD_FLOAT32_C(  -498.25), EASYSIMD_FLOAT32_C(  -825.81),
                         EASYSIMD_FLOAT32_C(  -382.30), EASYSIMD_FLOAT32_C(   749.76), EASYSIMD_FLOAT32_C(   -88.93), EASYSIMD_FLOAT32_C(  -767.88),
                         EASYSIMD_FLOAT32_C(   329.47), EASYSIMD_FLOAT32_C(  -783.86), EASYSIMD_FLOAT32_C(  -660.92), EASYSIMD_FLOAT32_C(   389.43));
  b = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -834.61), EASYSIMD_FLOAT32_C(   745.03), EASYSIMD_FLOAT32_C(   757.87), EASYSIMD_FLOAT32_C(  -224.03),
                         EASYSIMD_FLOAT32_C(  -773.34), EASYSIMD_FLOAT32_C(   -89.11), EASYSIMD_FLOAT32_C(  -807.38), EASYSIMD_FLOAT32_C(  -555.88),
                         EASYSIMD_FLOAT32_C(   155.24), EASYSIMD_FLOAT32_C(  -134.49), EASYSIMD_FLOAT32_C(   -64.47), EASYSIMD_FLOAT32_C(  -292.64),
                         EASYSIMD_FLOAT32_C(  -365.68), EASYSIMD_FLOAT32_C(  -507.71), EASYSIMD_FLOAT32_C(  -727.01), EASYSIMD_FLOAT32_C(   801.81));
  e = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -424.68), EASYSIMD_FLOAT32_C(  -312.49), EASYSIMD_FLOAT32_C(   755.66), EASYSIMD_FLOAT32_C(   224.03),
                         EASYSIMD_FLOAT32_C(  -494.18), EASYSIMD_FLOAT32_C(    89.11), EASYSIMD_FLOAT32_C(  -498.25), EASYSIMD_FLOAT32_C(  -556.76),
                         EASYSIMD_FLOAT32_C(  -565.63), EASYSIMD_FLOAT32_C(   134.49), EASYSIMD_FLOAT32_C(   -64.47), EASYSIMD_FLOAT32_C(  -292.64),
                         EASYSIMD_FLOAT32_C(   329.47), EASYSIMD_FLOAT32_C(  -326.81), EASYSIMD_FLOAT32_C(  -818.31), EASYSIMD_FLOAT32_C(   738.99));
  r = easysimd_mm512_mask_range_ps(src, UINT16_C(56952), a, b, INT32_C(           2));
  easysimd_test_x86_assert_equal_f32x16(r, e, 1);

  src = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -487.26), EASYSIMD_FLOAT32_C(  -990.41), EASYSIMD_FLOAT32_C(  -776.19), EASYSIMD_FLOAT32_C(  -760.25),
                           EASYSIMD_FLOAT32_C(  -792.22), EASYSIMD_FLOAT32_C(   648.49), EASYSIMD_FLOAT32_C(   552.24), EASYSIMD_FLOAT32_C(  -292.21),
                           EASYSIMD_FLOAT32_C(   746.37), EASYSIMD_FLOAT32_C(    46.42), EASYSIMD_FLOAT32_C(   -53.22), EASYSIMD_FLOAT32_C(   244.62),
                           EASYSIMD_FLOAT32_C(  -127.76), EASYSIMD_FLOAT32_C(  -670.92), EASYSIMD_FLOAT32_C(   494.86), EASYSIMD_FLOAT32_C(   961.17));
  a = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   470.13), EASYSIMD_FLOAT32_C(  -861.73), EASYSIMD_FLOAT32_C(  -971.39), EASYSIMD_FLOAT32_C(   141.05),
                         EASYSIMD_FLOAT32_C(  -356.60), EASYSIMD_FLOAT32_C(  -932.56), EASYSIMD_FLOAT32_C(    44.10), EASYSIMD_FLOAT32_C(  -521.99),
                         EASYSIMD_FLOAT32_C(  -677.60), EASYSIMD_FLOAT32_C(   286.22), EASYSIMD_FLOAT32_C(   702.04), EASYSIMD_FLOAT32_C(  -904.25),
                         EASYSIMD_FLOAT32_C(  -624.67), EASYSIMD_FLOAT32_C(   509.42), EASYSIMD_FLOAT32_C(   651.63), EASYSIMD_FLOAT32_C(   220.10));
  b = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   472.10), EASYSIMD_FLOAT32_C(   985.98), EASYSIMD_FLOAT32_C(   717.65), EASYSIMD_FLOAT32_C(  -748.00),
                         EASYSIMD_FLOAT32_C(   342.07), EASYSIMD_FLOAT32_C(     1.55), EASYSIMD_FLOAT32_C(   739.26), EASYSIMD_FLOAT32_C(   332.48),
                         EASYSIMD_FLOAT32_C(  -222.26), EASYSIMD_FLOAT32_C(   499.51), EASYSIMD_FLOAT32_C(   124.70), EASYSIMD_FLOAT32_C(   129.25),
                         EASYSIMD_FLOAT32_C(   947.26), EASYSIMD_FLOAT32_C(  -583.09), EASYSIMD_FLOAT32_C(   382.88), EASYSIMD_FLOAT32_C(   -99.16));
  e = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -487.26), EASYSIMD_FLOAT32_C(  -985.98), EASYSIMD_FLOAT32_C(  -776.19), EASYSIMD_FLOAT32_C(  -748.00),
                         EASYSIMD_FLOAT32_C(  -792.22), EASYSIMD_FLOAT32_C(  -932.56), EASYSIMD_FLOAT32_C(   552.24), EASYSIMD_FLOAT32_C(  -521.99),
                         EASYSIMD_FLOAT32_C(  -677.60), EASYSIMD_FLOAT32_C(  -499.51), EASYSIMD_FLOAT32_C(  -702.04), EASYSIMD_FLOAT32_C(  -904.25),
                         EASYSIMD_FLOAT32_C(  -127.76), EASYSIMD_FLOAT32_C(  -583.09), EASYSIMD_FLOAT32_C(   494.86), EASYSIMD_FLOAT32_C(  -220.10));
  r = easysimd_mm512_mask_range_ps(src, UINT16_C(22005), a, b, INT32_C(          15));
  easysimd_test_x86_assert_equal_f32x16(r, e, 1);

  src = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   840.01), EASYSIMD_FLOAT32_C(  -215.69), EASYSIMD_FLOAT32_C(  -211.07), EASYSIMD_FLOAT32_C(  -542.88),
                           EASYSIMD_FLOAT32_C(   883.47), EASYSIMD_FLOAT32_C(   318.80), EASYSIMD_FLOAT32_C(  -681.14), EASYSIMD_FLOAT32_C(   854.86),
                           EASYSIMD_FLOAT32_C(  -822.25), EASYSIMD_FLOAT32_C(   675.45), EASYSIMD_FLOAT32_C(   787.42), EASYSIMD_FLOAT32_C(   133.66),
                           EASYSIMD_FLOAT32_C(   197.45), EASYSIMD_FLOAT32_C(   465.02), EASYSIMD_FLOAT32_C(   847.43), EASYSIMD_FLOAT32_C(   495.40));
  a = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   300.48), EASYSIMD_FLOAT32_C(   541.87), EASYSIMD_FLOAT32_C(  -513.52), EASYSIMD_FLOAT32_C(   835.47),
                         EASYSIMD_FLOAT32_C(   694.44), EASYSIMD_FLOAT32_C(    -8.92), EASYSIMD_FLOAT32_C(  -533.81), EASYSIMD_FLOAT32_C(  -777.66),
                         EASYSIMD_FLOAT32_C(     5.10), EASYSIMD_FLOAT32_C(  -251.45), EASYSIMD_FLOAT32_C(   970.34), EASYSIMD_FLOAT32_C(   663.03),
                         EASYSIMD_FLOAT32_C(   747.00), EASYSIMD_FLOAT32_C(  -768.92), EASYSIMD_FLOAT32_C(  -669.45), EASYSIMD_FLOAT32_C(   -30.74));
  b = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   -55.37), EASYSIMD_FLOAT32_C(  -657.89), EASYSIMD_FLOAT32_C(  -833.14), EASYSIMD_FLOAT32_C(   975.37),
                         EASYSIMD_FLOAT32_C(   610.54), EASYSIMD_FLOAT32_C(   -38.98), EASYSIMD_FLOAT32_C(  -864.63), EASYSIMD_FLOAT32_C(  -173.77),
                         EASYSIMD_FLOAT32_C(  -827.92), EASYSIMD_FLOAT32_C(   678.24), EASYSIMD_FLOAT32_C(   -57.23), EASYSIMD_FLOAT32_C(  -146.72),
                         EASYSIMD_FLOAT32_C(   359.38), EASYSIMD_FLOAT32_C(    87.91), EASYSIMD_FLOAT32_C(  -324.47), EASYSIMD_FLOAT32_C(   683.93));
  e = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   840.01), EASYSIMD_FLOAT32_C(   541.87), EASYSIMD_FLOAT32_C(  -513.52), EASYSIMD_FLOAT32_C(  -542.88),
                         EASYSIMD_FLOAT32_C(   694.44), EASYSIMD_FLOAT32_C(   318.80), EASYSIMD_FLOAT32_C(  -681.14), EASYSIMD_FLOAT32_C(   854.86),
                         EASYSIMD_FLOAT32_C(     5.10), EASYSIMD_FLOAT32_C(  -678.24), EASYSIMD_FLOAT32_C(   787.42), EASYSIMD_FLOAT32_C(   663.03),
                         EASYSIMD_FLOAT32_C(   197.45), EASYSIMD_FLOAT32_C(   465.02), EASYSIMD_FLOAT32_C(  -324.47), EASYSIMD_FLOAT32_C(  -683.93));
  r = easysimd_mm512_mask_range_ps(src, UINT16_C(26835), a, b, INT32_C(           1));
  easysimd_test_x86_assert_equal_f32x16(r, e, 1);

  src = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -322.30), EASYSIMD_FLOAT32_C(  -672.98), EASYSIMD_FLOAT32_C(    42.32), EASYSIMD_FLOAT32_C(  -997.83),
                           EASYSIMD_FLOAT32_C(  -356.91), EASYSIMD_FLOAT32_C(   741.84), EASYSIMD_FLOAT32_C(  -539.70), EASYSIMD_FLOAT32_C(  -843.39),
                           EASYSIMD_FLOAT32_C(   906.37), EASYSIMD_FLOAT32_C(  -234.14), EASYSIMD_FLOAT32_C(   165.53), EASYSIMD_FLOAT32_C(   440.18),
                           EASYSIMD_FLOAT32_C(  -456.48), EASYSIMD_FLOAT32_C(  -839.57), EASYSIMD_FLOAT32_C(  -308.37), EASYSIMD_FLOAT32_C(  -426.81));
  a = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   488.78), EASYSIMD_FLOAT32_C(   111.65), EASYSIMD_FLOAT32_C(  -574.94), EASYSIMD_FLOAT32_C(   328.35),
                         EASYSIMD_FLOAT32_C(  -579.98), EASYSIMD_FLOAT32_C(   851.88), EASYSIMD_FLOAT32_C(  -169.06), EASYSIMD_FLOAT32_C(   475.39),
                         EASYSIMD_FLOAT32_C(   509.77), EASYSIMD_FLOAT32_C(  -335.92), EASYSIMD_FLOAT32_C(   500.01), EASYSIMD_FLOAT32_C(   899.23),
                         EASYSIMD_FLOAT32_C(   703.06), EASYSIMD_FLOAT32_C(   364.65), EASYSIMD_FLOAT32_C(    73.00), EASYSIMD_FLOAT32_C(   530.98));
  b = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -594.39), EASYSIMD_FLOAT32_C(   467.44), EASYSIMD_FLOAT32_C(   372.60), EASYSIMD_FLOAT32_C(  -125.37),
                         EASYSIMD_FLOAT32_C(   781.03), EASYSIMD_FLOAT32_C(   242.37), EASYSIMD_FLOAT32_C(  -803.07), EASYSIMD_FLOAT32_C(   454.01),
                         EASYSIMD_FLOAT32_C(  -799.96), EASYSIMD_FLOAT32_C(  -805.24), EASYSIMD_FLOAT32_C(  -189.08), EASYSIMD_FLOAT32_C(  -541.80),
                         EASYSIMD_FLOAT32_C(   734.45), EASYSIMD_FLOAT32_C(  -345.69), EASYSIMD_FLOAT32_C(  -448.17), EASYSIMD_FLOAT32_C(   -31.41));
  e = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -488.78), EASYSIMD_FLOAT32_C(  -467.44), EASYSIMD_FLOAT32_C(    42.32), EASYSIMD_FLOAT32_C(  -328.35),
                         EASYSIMD_FLOAT32_C(  -356.91), EASYSIMD_FLOAT32_C(   741.84), EASYSIMD_FLOAT32_C(  -539.70), EASYSIMD_FLOAT32_C(  -843.39),
                         EASYSIMD_FLOAT32_C(   906.37), EASYSIMD_FLOAT32_C(  -335.92), EASYSIMD_FLOAT32_C(   165.53), EASYSIMD_FLOAT32_C(   440.18),
                         EASYSIMD_FLOAT32_C(  -456.48), EASYSIMD_FLOAT32_C(  -364.65), EASYSIMD_FLOAT32_C(  -308.37), EASYSIMD_FLOAT32_C(  -426.81));
  r = easysimd_mm512_mask_range_ps(src, UINT16_C(53316), a, b, INT32_C(          13));
  easysimd_test_x86_assert_equal_f32x16(r, e, 1);

  src = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   890.99), EASYSIMD_FLOAT32_C(   100.13), EASYSIMD_FLOAT32_C(  -579.19), EASYSIMD_FLOAT32_C(   339.16),
                           EASYSIMD_FLOAT32_C(  -868.46), EASYSIMD_FLOAT32_C(   -67.97), EASYSIMD_FLOAT32_C(  -772.49), EASYSIMD_FLOAT32_C(   706.48),
                           EASYSIMD_FLOAT32_C(   603.69), EASYSIMD_FLOAT32_C(   807.49), EASYSIMD_FLOAT32_C(   854.60), EASYSIMD_FLOAT32_C(  -227.25),
                           EASYSIMD_FLOAT32_C(  -667.89), EASYSIMD_FLOAT32_C(  -655.17), EASYSIMD_FLOAT32_C(  -891.33), EASYSIMD_FLOAT32_C(  -167.91));
  a = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   954.63), EASYSIMD_FLOAT32_C(  -384.81), EASYSIMD_FLOAT32_C(   420.61), EASYSIMD_FLOAT32_C(   609.80),
                         EASYSIMD_FLOAT32_C(  -493.49), EASYSIMD_FLOAT32_C(  -411.48), EASYSIMD_FLOAT32_C(   164.20), EASYSIMD_FLOAT32_C(  -899.10),
                         EASYSIMD_FLOAT32_C(   121.08), EASYSIMD_FLOAT32_C(   791.60), EASYSIMD_FLOAT32_C(   226.27), EASYSIMD_FLOAT32_C(   340.05),
                         EASYSIMD_FLOAT32_C(  -450.77), EASYSIMD_FLOAT32_C(    29.34), EASYSIMD_FLOAT32_C(   886.04), EASYSIMD_FLOAT32_C(  -650.81));
  b = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   367.06), EASYSIMD_FLOAT32_C(   581.97), EASYSIMD_FLOAT32_C(   419.59), EASYSIMD_FLOAT32_C(    17.87),
                         EASYSIMD_FLOAT32_C(  -252.62), EASYSIMD_FLOAT32_C(  -655.53), EASYSIMD_FLOAT32_C(   126.88), EASYSIMD_FLOAT32_C(   647.25),
                         EASYSIMD_FLOAT32_C(   923.66), EASYSIMD_FLOAT32_C(   787.72), EASYSIMD_FLOAT32_C(   515.71), EASYSIMD_FLOAT32_C(    -8.38),
                         EASYSIMD_FLOAT32_C(   560.21), EASYSIMD_FLOAT32_C(   809.23), EASYSIMD_FLOAT32_C(   387.94), EASYSIMD_FLOAT32_C(   752.72));
  e = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   890.99), EASYSIMD_FLOAT32_C(   100.13), EASYSIMD_FLOAT32_C(  -579.19), EASYSIMD_FLOAT32_C(    17.87),
                         EASYSIMD_FLOAT32_C(  -868.46), EASYSIMD_FLOAT32_C(   -67.97), EASYSIMD_FLOAT32_C(   126.88), EASYSIMD_FLOAT32_C(   647.25),
                         EASYSIMD_FLOAT32_C(   121.08), EASYSIMD_FLOAT32_C(   787.72), EASYSIMD_FLOAT32_C(   854.60), EASYSIMD_FLOAT32_C(  -227.25),
                         EASYSIMD_FLOAT32_C(  -667.89), EASYSIMD_FLOAT32_C(    29.34), EASYSIMD_FLOAT32_C(   387.94), EASYSIMD_FLOAT32_C(   650.81));
  r = easysimd_mm512_mask_range_ps(src, UINT16_C( 5063), a, b, INT32_C(          10));
  easysimd_test_x86_assert_equal_f32x16(r, e, 1);

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 src = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    int imm8 = easysimd_test_codegen_rand() & 15;
    easysimd__m512 r;
    EASYSIMD_CONSTIFY_16_(easysimd_mm512_mask_range_ps, r, easysimd_mm512_setzero_ps(), imm8, src, k, a, b);

    easysimd_test_x86_write_f32x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_range_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  easysimd__m512 a, b, e, r;

  a = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    88.80), EASYSIMD_FLOAT32_C(  -754.26), EASYSIMD_FLOAT32_C(  -551.88), EASYSIMD_FLOAT32_C(   528.59),
                         EASYSIMD_FLOAT32_C(  -563.49), EASYSIMD_FLOAT32_C(    60.18), EASYSIMD_FLOAT32_C(   775.88), EASYSIMD_FLOAT32_C(  -518.12),
                         EASYSIMD_FLOAT32_C(  -555.00), EASYSIMD_FLOAT32_C(  -644.73), EASYSIMD_FLOAT32_C(  -127.91), EASYSIMD_FLOAT32_C(   938.48),
                         EASYSIMD_FLOAT32_C(   766.75), EASYSIMD_FLOAT32_C(   707.89), EASYSIMD_FLOAT32_C(   837.58), EASYSIMD_FLOAT32_C(  -354.33));
  b = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(    78.74), EASYSIMD_FLOAT32_C(  -695.37), EASYSIMD_FLOAT32_C(  -650.65), EASYSIMD_FLOAT32_C(  -566.92),
                         EASYSIMD_FLOAT32_C(  -611.66), EASYSIMD_FLOAT32_C(   738.04), EASYSIMD_FLOAT32_C(   127.45), EASYSIMD_FLOAT32_C(    21.28),
                         EASYSIMD_FLOAT32_C(  -843.93), EASYSIMD_FLOAT32_C(   707.87), EASYSIMD_FLOAT32_C(  -996.59), EASYSIMD_FLOAT32_C(   408.69),
                         EASYSIMD_FLOAT32_C(   363.40), EASYSIMD_FLOAT32_C(  -123.48), EASYSIMD_FLOAT32_C(   761.44), EASYSIMD_FLOAT32_C(   439.74));
  e = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -695.37), EASYSIMD_FLOAT32_C(  -551.88), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -738.04), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(  -555.00), EASYSIMD_FLOAT32_C(  -707.87), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(  -766.75), EASYSIMD_FLOAT32_C(  -707.89), EASYSIMD_FLOAT32_C(  -837.58), EASYSIMD_FLOAT32_C(  -439.74));
  r = easysimd_mm512_maskz_range_ps(UINT16_C(25807), a, b, INT32_C(          13));
  easysimd_test_x86_assert_equal_f32x16(r, e, 1);

  a = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -118.14), EASYSIMD_FLOAT32_C(  -529.44), EASYSIMD_FLOAT32_C(   810.18), EASYSIMD_FLOAT32_C(   518.46),
                         EASYSIMD_FLOAT32_C(   594.04), EASYSIMD_FLOAT32_C(  -951.27), EASYSIMD_FLOAT32_C(  -921.28), EASYSIMD_FLOAT32_C(  -494.77),
                         EASYSIMD_FLOAT32_C(   803.00), EASYSIMD_FLOAT32_C(   630.60), EASYSIMD_FLOAT32_C(   -23.36), EASYSIMD_FLOAT32_C(   366.49),
                         EASYSIMD_FLOAT32_C(  -429.58), EASYSIMD_FLOAT32_C(   200.76), EASYSIMD_FLOAT32_C(  -115.40), EASYSIMD_FLOAT32_C(  -874.58));
  b = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   774.70), EASYSIMD_FLOAT32_C(  -925.51), EASYSIMD_FLOAT32_C(  -799.28), EASYSIMD_FLOAT32_C(   649.28),
                         EASYSIMD_FLOAT32_C(   229.00), EASYSIMD_FLOAT32_C(  -811.80), EASYSIMD_FLOAT32_C(   462.34), EASYSIMD_FLOAT32_C(  -849.74),
                         EASYSIMD_FLOAT32_C(   883.58), EASYSIMD_FLOAT32_C(   112.99), EASYSIMD_FLOAT32_C(   717.18), EASYSIMD_FLOAT32_C(   495.24),
                         EASYSIMD_FLOAT32_C(   374.94), EASYSIMD_FLOAT32_C(  -410.27), EASYSIMD_FLOAT32_C(  -526.03), EASYSIMD_FLOAT32_C(   218.87));
  e = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   529.44), EASYSIMD_FLOAT32_C(   810.18), EASYSIMD_FLOAT32_C(   649.28),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   630.60), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   495.24),
                         EASYSIMD_FLOAT32_C(   374.94), EASYSIMD_FLOAT32_C(   200.76), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00));
  r = easysimd_mm512_maskz_range_ps(UINT16_C(28764), a, b, INT32_C(           9));
  easysimd_test_x86_assert_equal_f32x16(r, e, 1);

  a = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -292.48), EASYSIMD_FLOAT32_C(    44.49), EASYSIMD_FLOAT32_C(   295.69), EASYSIMD_FLOAT32_C(   332.58),
                         EASYSIMD_FLOAT32_C(  -545.24), EASYSIMD_FLOAT32_C(  -178.28), EASYSIMD_FLOAT32_C(  -886.29), EASYSIMD_FLOAT32_C(   572.90),
                         EASYSIMD_FLOAT32_C(  -648.84), EASYSIMD_FLOAT32_C(  -696.46), EASYSIMD_FLOAT32_C(  -945.56), EASYSIMD_FLOAT32_C(  -242.87),
                         EASYSIMD_FLOAT32_C(  -745.19), EASYSIMD_FLOAT32_C(   975.72), EASYSIMD_FLOAT32_C(  -748.11), EASYSIMD_FLOAT32_C(  -548.19));
  b = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -399.44), EASYSIMD_FLOAT32_C(    51.94), EASYSIMD_FLOAT32_C(   490.68), EASYSIMD_FLOAT32_C(  -851.25),
                         EASYSIMD_FLOAT32_C(  -293.18), EASYSIMD_FLOAT32_C(  -784.57), EASYSIMD_FLOAT32_C(    63.42), EASYSIMD_FLOAT32_C(   -67.87),
                         EASYSIMD_FLOAT32_C(  -859.07), EASYSIMD_FLOAT32_C(  -137.29), EASYSIMD_FLOAT32_C(   282.85), EASYSIMD_FLOAT32_C(   -88.07),
                         EASYSIMD_FLOAT32_C(  -325.50), EASYSIMD_FLOAT32_C(   820.51), EASYSIMD_FLOAT32_C(  -238.33), EASYSIMD_FLOAT32_C(  -209.07));
  e = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    44.49), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -178.28), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -137.29), EASYSIMD_FLOAT32_C(   282.85), EASYSIMD_FLOAT32_C(   -88.07),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   820.51), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00));
  r = easysimd_mm512_maskz_range_ps(UINT16_C(17524), a, b, INT32_C(           6));
  easysimd_test_x86_assert_equal_f32x16(r, e, 1);

  a = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -566.29), EASYSIMD_FLOAT32_C(   133.23), EASYSIMD_FLOAT32_C(   -84.08), EASYSIMD_FLOAT32_C(   759.21),
                         EASYSIMD_FLOAT32_C(   312.72), EASYSIMD_FLOAT32_C(  -845.75), EASYSIMD_FLOAT32_C(   -31.72), EASYSIMD_FLOAT32_C(  -394.80),
                         EASYSIMD_FLOAT32_C(   109.76), EASYSIMD_FLOAT32_C(   672.59), EASYSIMD_FLOAT32_C(   272.61), EASYSIMD_FLOAT32_C(  -345.01),
                         EASYSIMD_FLOAT32_C(  -149.13), EASYSIMD_FLOAT32_C(   158.90), EASYSIMD_FLOAT32_C(    82.09), EASYSIMD_FLOAT32_C(  -500.29));
  b = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   917.18), EASYSIMD_FLOAT32_C(   964.51), EASYSIMD_FLOAT32_C(  -865.38), EASYSIMD_FLOAT32_C(   417.47),
                         EASYSIMD_FLOAT32_C(   109.15), EASYSIMD_FLOAT32_C(   106.96), EASYSIMD_FLOAT32_C(   674.90), EASYSIMD_FLOAT32_C(  -491.41),
                         EASYSIMD_FLOAT32_C(  -944.97), EASYSIMD_FLOAT32_C(  -815.79), EASYSIMD_FLOAT32_C(  -640.16), EASYSIMD_FLOAT32_C(   348.21),
                         EASYSIMD_FLOAT32_C(   968.78), EASYSIMD_FLOAT32_C(   296.42), EASYSIMD_FLOAT32_C(  -583.92), EASYSIMD_FLOAT32_C(   827.85));
  e = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -566.29), EASYSIMD_FLOAT32_C(  -133.23), EASYSIMD_FLOAT32_C(  -865.38), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(  -109.15), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -640.16), EASYSIMD_FLOAT32_C(  -345.01),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -500.29));
  r = easysimd_mm512_maskz_range_ps(UINT16_C(59441), a, b, INT32_C(          12));
  easysimd_test_x86_assert_equal_f32x16(r, e, 1);

  a = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   848.26), EASYSIMD_FLOAT32_C(  -101.73), EASYSIMD_FLOAT32_C(   863.25), EASYSIMD_FLOAT32_C(   879.47),
                         EASYSIMD_FLOAT32_C(   601.85), EASYSIMD_FLOAT32_C(   447.17), EASYSIMD_FLOAT32_C(  -948.38), EASYSIMD_FLOAT32_C(   168.13),
                         EASYSIMD_FLOAT32_C(  -686.06), EASYSIMD_FLOAT32_C(   135.70), EASYSIMD_FLOAT32_C(   408.92), EASYSIMD_FLOAT32_C(     1.22),
                         EASYSIMD_FLOAT32_C(   -18.55), EASYSIMD_FLOAT32_C(  -559.36), EASYSIMD_FLOAT32_C(  -603.98), EASYSIMD_FLOAT32_C(   871.70));
  b = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   596.47), EASYSIMD_FLOAT32_C(   810.07), EASYSIMD_FLOAT32_C(   963.76), EASYSIMD_FLOAT32_C(   724.77),
                         EASYSIMD_FLOAT32_C(    42.02), EASYSIMD_FLOAT32_C(  -159.64), EASYSIMD_FLOAT32_C(  -491.94), EASYSIMD_FLOAT32_C(   124.84),
                         EASYSIMD_FLOAT32_C(  -124.15), EASYSIMD_FLOAT32_C(  -626.55), EASYSIMD_FLOAT32_C(   707.37), EASYSIMD_FLOAT32_C(   766.70),
                         EASYSIMD_FLOAT32_C(   266.48), EASYSIMD_FLOAT32_C(  -967.53), EASYSIMD_FLOAT32_C(   258.11), EASYSIMD_FLOAT32_C(   211.45));
  e = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   596.47), EASYSIMD_FLOAT32_C(  -101.73), EASYSIMD_FLOAT32_C(   863.25), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -491.94), EASYSIMD_FLOAT32_C(   124.84),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(   -18.55), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   211.45));
  r = easysimd_mm512_maskz_range_ps(UINT16_C(58121), a, b, INT32_C(           2));
  easysimd_test_x86_assert_equal_f32x16(r, e, 1);

  a = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   463.30), EASYSIMD_FLOAT32_C(   525.45), EASYSIMD_FLOAT32_C(  -414.02), EASYSIMD_FLOAT32_C(  -803.19),
                         EASYSIMD_FLOAT32_C(   492.98), EASYSIMD_FLOAT32_C(   327.87), EASYSIMD_FLOAT32_C(   -14.64), EASYSIMD_FLOAT32_C(   644.72),
                         EASYSIMD_FLOAT32_C(  -570.39), EASYSIMD_FLOAT32_C(   122.11), EASYSIMD_FLOAT32_C(   765.25), EASYSIMD_FLOAT32_C(  -172.24),
                         EASYSIMD_FLOAT32_C(   674.94), EASYSIMD_FLOAT32_C(   713.63), EASYSIMD_FLOAT32_C(   659.63), EASYSIMD_FLOAT32_C(   361.00));
  b = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   753.45), EASYSIMD_FLOAT32_C(  -756.03), EASYSIMD_FLOAT32_C(   460.45), EASYSIMD_FLOAT32_C(  -607.56),
                         EASYSIMD_FLOAT32_C(   666.05), EASYSIMD_FLOAT32_C(   209.74), EASYSIMD_FLOAT32_C(  -967.34), EASYSIMD_FLOAT32_C(  -930.42),
                         EASYSIMD_FLOAT32_C(   399.68), EASYSIMD_FLOAT32_C(  -931.11), EASYSIMD_FLOAT32_C(  -655.19), EASYSIMD_FLOAT32_C(  -642.34),
                         EASYSIMD_FLOAT32_C(   228.53), EASYSIMD_FLOAT32_C(   836.74), EASYSIMD_FLOAT32_C(   232.82), EASYSIMD_FLOAT32_C(  -647.32));
  e = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   525.45), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   607.56),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   327.87), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(   399.68), EASYSIMD_FLOAT32_C(   122.11), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   172.24),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   836.74), EASYSIMD_FLOAT32_C(   659.63), EASYSIMD_FLOAT32_C(     0.00));
  r = easysimd_mm512_maskz_range_ps(UINT16_C(21718), a, b, INT32_C(           9));
  easysimd_test_x86_assert_equal_f32x16(r, e, 1);

  a = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   872.53), EASYSIMD_FLOAT32_C(    32.71), EASYSIMD_FLOAT32_C(   618.82), EASYSIMD_FLOAT32_C(  -356.01),
                         EASYSIMD_FLOAT32_C(   195.97), EASYSIMD_FLOAT32_C(  -613.99), EASYSIMD_FLOAT32_C(  -708.69), EASYSIMD_FLOAT32_C(   732.67),
                         EASYSIMD_FLOAT32_C(  -139.44), EASYSIMD_FLOAT32_C(   705.33), EASYSIMD_FLOAT32_C(   535.86), EASYSIMD_FLOAT32_C(   367.58),
                         EASYSIMD_FLOAT32_C(  -622.55), EASYSIMD_FLOAT32_C(  -449.50), EASYSIMD_FLOAT32_C(   722.85), EASYSIMD_FLOAT32_C(   947.85));
  b = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   114.29), EASYSIMD_FLOAT32_C(   548.35), EASYSIMD_FLOAT32_C(   314.72), EASYSIMD_FLOAT32_C(   166.44),
                         EASYSIMD_FLOAT32_C(  -880.04), EASYSIMD_FLOAT32_C(   357.12), EASYSIMD_FLOAT32_C(  -953.64), EASYSIMD_FLOAT32_C(  -633.48),
                         EASYSIMD_FLOAT32_C(   113.14), EASYSIMD_FLOAT32_C(  -414.10), EASYSIMD_FLOAT32_C(   974.07), EASYSIMD_FLOAT32_C(   447.09),
                         EASYSIMD_FLOAT32_C(   376.16), EASYSIMD_FLOAT32_C(   941.42), EASYSIMD_FLOAT32_C(   377.52), EASYSIMD_FLOAT32_C(   976.48));
  e = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   548.35), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -166.44),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -357.12), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   732.67),
                         EASYSIMD_FLOAT32_C(  -113.14), EASYSIMD_FLOAT32_C(   705.33), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   447.09),
                         EASYSIMD_FLOAT32_C(  -376.16), EASYSIMD_FLOAT32_C(  -941.42), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   976.48));
  r = easysimd_mm512_maskz_range_ps(UINT16_C(21981), a, b, INT32_C(           1));
  easysimd_test_x86_assert_equal_f32x16(r, e, 1);

  a = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(  -376.83), EASYSIMD_FLOAT32_C(   946.31), EASYSIMD_FLOAT32_C(   -26.41), EASYSIMD_FLOAT32_C(   247.02),
                         EASYSIMD_FLOAT32_C(  -995.11), EASYSIMD_FLOAT32_C(   596.07), EASYSIMD_FLOAT32_C(   270.54), EASYSIMD_FLOAT32_C(  -867.63),
                         EASYSIMD_FLOAT32_C(  -436.64), EASYSIMD_FLOAT32_C(   651.71), EASYSIMD_FLOAT32_C(   488.37), EASYSIMD_FLOAT32_C(   367.39),
                         EASYSIMD_FLOAT32_C(   265.70), EASYSIMD_FLOAT32_C(   197.07), EASYSIMD_FLOAT32_C(   634.72), EASYSIMD_FLOAT32_C(  -594.85));
  b = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(   648.39), EASYSIMD_FLOAT32_C(    27.90), EASYSIMD_FLOAT32_C(  -945.93), EASYSIMD_FLOAT32_C(   243.24),
                         EASYSIMD_FLOAT32_C(   536.16), EASYSIMD_FLOAT32_C(   955.21), EASYSIMD_FLOAT32_C(  -794.33), EASYSIMD_FLOAT32_C(  -578.13),
                         EASYSIMD_FLOAT32_C(  -593.14), EASYSIMD_FLOAT32_C(  -109.05), EASYSIMD_FLOAT32_C(   255.43), EASYSIMD_FLOAT32_C(  -713.10),
                         EASYSIMD_FLOAT32_C(   533.83), EASYSIMD_FLOAT32_C(   209.07), EASYSIMD_FLOAT32_C(   920.38), EASYSIMD_FLOAT32_C(  -579.31));
  e = easysimd_mm512_set_ps(EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   -26.41), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   270.54), EASYSIMD_FLOAT32_C(     0.00),
                         EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   109.05), EASYSIMD_FLOAT32_C(   255.43), EASYSIMD_FLOAT32_C(   367.39),
                         EASYSIMD_FLOAT32_C(   265.70), EASYSIMD_FLOAT32_C(   197.07), EASYSIMD_FLOAT32_C(   634.72), EASYSIMD_FLOAT32_C(  -579.31));
  r = easysimd_mm512_maskz_range_ps(UINT16_C( 8831), a, b, INT32_C(           2));
  easysimd_test_x86_assert_equal_f32x16(r, e, 1);

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 b = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    int imm8 = easysimd_test_codegen_rand() & 15;
    easysimd__m512 r;
    EASYSIMD_CONSTIFY_16_(easysimd_mm512_maskz_range_ps, r, easysimd_mm512_setzero_ps(), imm8, k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_range_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   830.67), EASYSIMD_FLOAT64_C(   598.43) },
      { EASYSIMD_FLOAT64_C(  -161.84), EASYSIMD_FLOAT64_C(  -609.24) },
      { EASYSIMD_FLOAT64_C(   830.67), EASYSIMD_FLOAT64_C(   609.24) } },
    { { EASYSIMD_FLOAT64_C(  -639.95), EASYSIMD_FLOAT64_C(   257.59) },
      { EASYSIMD_FLOAT64_C(   -49.35), EASYSIMD_FLOAT64_C(   500.73) },
      { EASYSIMD_FLOAT64_C(  -639.95), EASYSIMD_FLOAT64_C(   257.59) } },
    { { EASYSIMD_FLOAT64_C(   920.50), EASYSIMD_FLOAT64_C(  -952.84) },
      { EASYSIMD_FLOAT64_C(   445.89), EASYSIMD_FLOAT64_C(  -703.72) },
      { EASYSIMD_FLOAT64_C(   920.50), EASYSIMD_FLOAT64_C(  -703.72) } },
    { { EASYSIMD_FLOAT64_C(   384.42), EASYSIMD_FLOAT64_C(  -127.63) },
      { EASYSIMD_FLOAT64_C(   800.45), EASYSIMD_FLOAT64_C(   678.44) },
      { EASYSIMD_FLOAT64_C(   384.42), EASYSIMD_FLOAT64_C(  -127.63) } },
    { { EASYSIMD_FLOAT64_C(   519.71), EASYSIMD_FLOAT64_C(   275.20) },
      { EASYSIMD_FLOAT64_C(  -380.64), EASYSIMD_FLOAT64_C(  -709.08) },
      { EASYSIMD_FLOAT64_C(   519.71), EASYSIMD_FLOAT64_C(   709.08) } },
    { { EASYSIMD_FLOAT64_C(  -162.04), EASYSIMD_FLOAT64_C(   472.77) },
      { EASYSIMD_FLOAT64_C(  -553.19), EASYSIMD_FLOAT64_C(   126.94) },
      { EASYSIMD_FLOAT64_C(   162.04), EASYSIMD_FLOAT64_C(   472.77) } },
    { { EASYSIMD_FLOAT64_C(  -894.91), EASYSIMD_FLOAT64_C(  -295.68) },
      { EASYSIMD_FLOAT64_C(   576.25), EASYSIMD_FLOAT64_C(   294.47) },
      { EASYSIMD_FLOAT64_C(   576.25), EASYSIMD_FLOAT64_C(   294.47) } },
    { { EASYSIMD_FLOAT64_C(    69.59), EASYSIMD_FLOAT64_C(  -855.72) },
      { EASYSIMD_FLOAT64_C(   780.93), EASYSIMD_FLOAT64_C(   -99.75) },
      { EASYSIMD_FLOAT64_C(    69.59), EASYSIMD_FLOAT64_C(  -855.72) } }
  };

  easysimd__m128d a, b, r;

  a = easysimd_mm_loadu_pd(test_vec[0].a);
  b = easysimd_mm_loadu_pd(test_vec[0].b);
  r = easysimd_mm_range_pd(a, b, 11);
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[1].a);
  b = easysimd_mm_loadu_pd(test_vec[1].b);
  r = easysimd_mm_range_pd(a, b, 4);
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[2].a);
  b = easysimd_mm_loadu_pd(test_vec[2].b);
  r = easysimd_mm_range_pd(a, b, 1);
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[3].a);
  b = easysimd_mm_loadu_pd(test_vec[3].b);
  r = easysimd_mm_range_pd(a, b, 2);
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[4].a);
  b = easysimd_mm_loadu_pd(test_vec[4].b);
  r = easysimd_mm_range_pd(a, b, 3);
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[5].a);
  b = easysimd_mm_loadu_pd(test_vec[5].b);
  r = easysimd_mm_range_pd(a, b, 9);
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[6].a);
  b = easysimd_mm_loadu_pd(test_vec[6].b);
  r = easysimd_mm_range_pd(a, b, 5);
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[7].a);
  b = easysimd_mm_loadu_pd(test_vec[7].b);
  r = easysimd_mm_range_pd(a, b, 0);
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);

  easysimd__m128d a, b, r;

  a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm_range_pd(a, b, 11);

  easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm_range_pd(a, b, 4);

  easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm_range_pd(a, b, 1);

  easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm_range_pd(a, b, 2);

  easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm_range_pd(a, b, 3);

  easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm_range_pd(a, b, 9);

  easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm_range_pd(a, b, 5);

  easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm_range_pd(a, b, 0);

  easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  return 1;
#endif
}

static int
test_easysimd_mm_mask_range_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  easysimd__m128d src, a, b, e, r;

  src = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -139.94), EASYSIMD_FLOAT64_C(  -886.75));
  a = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -792.78), EASYSIMD_FLOAT64_C(  -894.29));
  b = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -806.31), EASYSIMD_FLOAT64_C(   453.92));
  e = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -139.94), EASYSIMD_FLOAT64_C(  -894.29));
  r = easysimd_mm_mask_range_pd(src, UINT8_C(141), a, b, INT32_C(          12));
  easysimd_test_x86_assert_equal_f64x2(r, e, 1);

  src = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -663.72), EASYSIMD_FLOAT64_C(   184.82));
  a = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -194.35), EASYSIMD_FLOAT64_C(   403.49));
  b = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   681.58), EASYSIMD_FLOAT64_C(   390.93));
  e = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -663.72), EASYSIMD_FLOAT64_C(   403.49));
  r = easysimd_mm_mask_range_pd(src, UINT8_C( 93), a, b, INT32_C(          11));
  easysimd_test_x86_assert_equal_f64x2(r, e, 1);

  src = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -113.27), EASYSIMD_FLOAT64_C(  -276.87));
  a = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   963.08), EASYSIMD_FLOAT64_C(  -621.87));
  b = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   150.21), EASYSIMD_FLOAT64_C(   955.33));
  e = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -113.27), EASYSIMD_FLOAT64_C(  -621.87));
  r = easysimd_mm_mask_range_pd(src, UINT8_C(241), a, b, INT32_C(           4));
  easysimd_test_x86_assert_equal_f64x2(r, e, 1);

  src = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   985.23), EASYSIMD_FLOAT64_C(   499.81));
  a = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -193.24), EASYSIMD_FLOAT64_C(  -403.55));
  b = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   444.77), EASYSIMD_FLOAT64_C(  -416.56));
  e = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -444.77), EASYSIMD_FLOAT64_C(  -403.55));
  r = easysimd_mm_mask_range_pd(src, UINT8_C( 79), a, b, INT32_C(           1));
  easysimd_test_x86_assert_equal_f64x2(r, e, 1);

  src = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -695.93), EASYSIMD_FLOAT64_C(   443.50));
  a = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   757.98), EASYSIMD_FLOAT64_C(   650.72));
  b = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -329.26), EASYSIMD_FLOAT64_C(   219.41));
  e = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   757.98), EASYSIMD_FLOAT64_C(   650.72));
  r = easysimd_mm_mask_range_pd(src, UINT8_C( 87), a, b, INT32_C(           1));
  easysimd_test_x86_assert_equal_f64x2(r, e, 1);

  src = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   426.75), EASYSIMD_FLOAT64_C(   555.69));
  a = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -182.32), EASYSIMD_FLOAT64_C(  -638.66));
  b = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   952.11), EASYSIMD_FLOAT64_C(  -972.12));
  e = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -182.32), EASYSIMD_FLOAT64_C(   555.69));
  r = easysimd_mm_mask_range_pd(src, UINT8_C(182), a, b, INT32_C(           0));
  easysimd_test_x86_assert_equal_f64x2(r, e, 1);

  src = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    -2.92), EASYSIMD_FLOAT64_C(   -85.39));
  a = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   -47.59), EASYSIMD_FLOAT64_C(  -122.31));
  b = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   877.42), EASYSIMD_FLOAT64_C(    69.15));
  e = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   -47.59), EASYSIMD_FLOAT64_C(   -69.15));
  r = easysimd_mm_mask_range_pd(src, UINT8_C(143), a, b, INT32_C(           2));
  easysimd_test_x86_assert_equal_f64x2(r, e, 1);

  src = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -613.32), EASYSIMD_FLOAT64_C(    54.38));
  a = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   -29.88), EASYSIMD_FLOAT64_C(   861.14));
  b = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -218.85), EASYSIMD_FLOAT64_C(  -506.57));
  e = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -613.32), EASYSIMD_FLOAT64_C(   506.57));
  r = easysimd_mm_mask_range_pd(src, UINT8_C( 49), a, b, INT32_C(           8));
  easysimd_test_x86_assert_equal_f64x2(r, e, 1);

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d src = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    int imm8 = easysimd_test_codegen_rand() & 15;
    easysimd__m128d r;
    EASYSIMD_CONSTIFY_16_(easysimd_mm_mask_range_pd, r, easysimd_mm_setzero_pd(), imm8, src, k, a, b);

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
test_easysimd_mm_maskz_range_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  easysimd__m128d a, b, e, r;

  a = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -935.65), EASYSIMD_FLOAT64_C(   806.87));
  b = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    26.29), EASYSIMD_FLOAT64_C(  -444.52));
  e = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00));
  r = easysimd_mm_maskz_range_pd(UINT8_C(108), a, b, INT32_C(          14));
  easysimd_test_x86_assert_equal_f64x2(r, e, 1);

  a = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -838.16), EASYSIMD_FLOAT64_C(  -418.02));
  b = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   -56.68), EASYSIMD_FLOAT64_C(   844.57));
  e = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   844.57));
  r = easysimd_mm_maskz_range_pd(UINT8_C( 37), a, b, INT32_C(           9));
  easysimd_test_x86_assert_equal_f64x2(r, e, 1);

  a = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -479.67), EASYSIMD_FLOAT64_C(  -104.57));
  b = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   892.51), EASYSIMD_FLOAT64_C(  -212.94));
  e = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   892.51), EASYSIMD_FLOAT64_C(   104.57));
  r = easysimd_mm_maskz_range_pd(   UINT8_MAX, a, b, INT32_C(           9));
  easysimd_test_x86_assert_equal_f64x2(r, e, 1);

  a = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   508.41), EASYSIMD_FLOAT64_C(  -155.08));
  b = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   297.13), EASYSIMD_FLOAT64_C(   542.17));
  e = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   508.41), EASYSIMD_FLOAT64_C(   542.17));
  r = easysimd_mm_maskz_range_pd(UINT8_C(  3), a, b, INT32_C(           7));
  easysimd_test_x86_assert_equal_f64x2(r, e, 1);

  a = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -576.07), EASYSIMD_FLOAT64_C(  -654.20));
  b = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -160.77), EASYSIMD_FLOAT64_C(  -101.02));
  e = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -576.07), EASYSIMD_FLOAT64_C(  -654.20));
  r = easysimd_mm_maskz_range_pd(UINT8_C(187), a, b, INT32_C(           7));
  easysimd_test_x86_assert_equal_f64x2(r, e, 1);

  a = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(    11.95), EASYSIMD_FLOAT64_C(   636.72));
  b = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -807.80), EASYSIMD_FLOAT64_C(   376.95));
  e = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   -11.95), EASYSIMD_FLOAT64_C(  -376.95));
  r = easysimd_mm_maskz_range_pd(UINT8_C(211), a, b, INT32_C(          14));
  easysimd_test_x86_assert_equal_f64x2(r, e, 1);

  a = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -379.78), EASYSIMD_FLOAT64_C(   690.48));
  b = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   535.05), EASYSIMD_FLOAT64_C(  -726.12));
  e = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(   379.78), EASYSIMD_FLOAT64_C(   690.48));
  r = easysimd_mm_maskz_range_pd(UINT8_C(167), a, b, INT32_C(          10));
  easysimd_test_x86_assert_equal_f64x2(r, e, 1);

  a = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -541.03), EASYSIMD_FLOAT64_C(   407.50));
  b = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(  -805.43), EASYSIMD_FLOAT64_C(   773.72));
  e = easysimd_mm_set_pd(EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   407.50));
  r = easysimd_mm_maskz_range_pd(UINT8_C( 53), a, b, INT32_C(           6));
  easysimd_test_x86_assert_equal_f64x2(r, e, 1);

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    int imm8 = easysimd_test_codegen_rand() & 15;
    easysimd__m128d r;
    EASYSIMD_CONSTIFY_16_(easysimd_mm_maskz_range_pd, r, easysimd_mm_setzero_pd(), imm8, k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_range_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[4];
    const easysimd_float64 b[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   742.72), EASYSIMD_FLOAT64_C(  -380.91), EASYSIMD_FLOAT64_C(   291.01), EASYSIMD_FLOAT64_C(  -897.24) },
      { EASYSIMD_FLOAT64_C(   876.68), EASYSIMD_FLOAT64_C(  -758.34), EASYSIMD_FLOAT64_C(   603.49), EASYSIMD_FLOAT64_C(   797.18) },
      { EASYSIMD_FLOAT64_C(   876.68), EASYSIMD_FLOAT64_C(   758.34), EASYSIMD_FLOAT64_C(   603.49), EASYSIMD_FLOAT64_C(   897.24) } },
    { { EASYSIMD_FLOAT64_C(  -711.18), EASYSIMD_FLOAT64_C(    49.38), EASYSIMD_FLOAT64_C(  -906.54), EASYSIMD_FLOAT64_C(   673.24) },
      { EASYSIMD_FLOAT64_C(   921.74), EASYSIMD_FLOAT64_C(   893.90), EASYSIMD_FLOAT64_C(   351.69), EASYSIMD_FLOAT64_C(   441.45) },
      { EASYSIMD_FLOAT64_C(  -711.18), EASYSIMD_FLOAT64_C(    49.38), EASYSIMD_FLOAT64_C(  -906.54), EASYSIMD_FLOAT64_C(   441.45) } },
    { { EASYSIMD_FLOAT64_C(   169.11), EASYSIMD_FLOAT64_C(   971.04), EASYSIMD_FLOAT64_C(   732.37), EASYSIMD_FLOAT64_C(  -992.93) },
      { EASYSIMD_FLOAT64_C(   443.81), EASYSIMD_FLOAT64_C(  -820.82), EASYSIMD_FLOAT64_C(   134.01), EASYSIMD_FLOAT64_C(   548.91) },
      { EASYSIMD_FLOAT64_C(   443.81), EASYSIMD_FLOAT64_C(   971.04), EASYSIMD_FLOAT64_C(   732.37), EASYSIMD_FLOAT64_C(  -548.91) } },
    { { EASYSIMD_FLOAT64_C(  -116.50), EASYSIMD_FLOAT64_C(  -289.74), EASYSIMD_FLOAT64_C(  -156.63), EASYSIMD_FLOAT64_C(   953.09) },
      { EASYSIMD_FLOAT64_C(  -145.46), EASYSIMD_FLOAT64_C(  -375.70), EASYSIMD_FLOAT64_C(  -146.66), EASYSIMD_FLOAT64_C(  -402.75) },
      { EASYSIMD_FLOAT64_C(  -116.50), EASYSIMD_FLOAT64_C(  -289.74), EASYSIMD_FLOAT64_C(  -146.66), EASYSIMD_FLOAT64_C(   402.75) } },
    { { EASYSIMD_FLOAT64_C(   243.39), EASYSIMD_FLOAT64_C(  -855.65), EASYSIMD_FLOAT64_C(  -299.98), EASYSIMD_FLOAT64_C(   120.07) },
      { EASYSIMD_FLOAT64_C(  -613.99), EASYSIMD_FLOAT64_C(  -696.49), EASYSIMD_FLOAT64_C(   -82.74), EASYSIMD_FLOAT64_C(  -325.17) },
      { EASYSIMD_FLOAT64_C(   613.99), EASYSIMD_FLOAT64_C(  -855.65), EASYSIMD_FLOAT64_C(  -299.98), EASYSIMD_FLOAT64_C(   325.17) } },
    { { EASYSIMD_FLOAT64_C(   352.88), EASYSIMD_FLOAT64_C(    10.71), EASYSIMD_FLOAT64_C(  -651.93), EASYSIMD_FLOAT64_C(   274.62) },
      { EASYSIMD_FLOAT64_C(   -95.38), EASYSIMD_FLOAT64_C(   699.76), EASYSIMD_FLOAT64_C(  -283.92), EASYSIMD_FLOAT64_C(  -926.28) },
      { EASYSIMD_FLOAT64_C(   352.88), EASYSIMD_FLOAT64_C(   699.76), EASYSIMD_FLOAT64_C(   283.92), EASYSIMD_FLOAT64_C(   274.62) } },
    { { EASYSIMD_FLOAT64_C(   670.80), EASYSIMD_FLOAT64_C(  -551.55), EASYSIMD_FLOAT64_C(  -919.21), EASYSIMD_FLOAT64_C(   114.61) },
      { EASYSIMD_FLOAT64_C(  -372.37), EASYSIMD_FLOAT64_C(   214.80), EASYSIMD_FLOAT64_C(  -336.48), EASYSIMD_FLOAT64_C(   511.13) },
      { EASYSIMD_FLOAT64_C(   670.80), EASYSIMD_FLOAT64_C(   214.80), EASYSIMD_FLOAT64_C(  -336.48), EASYSIMD_FLOAT64_C(   511.13) } },
    { { EASYSIMD_FLOAT64_C(   925.05), EASYSIMD_FLOAT64_C(   506.89), EASYSIMD_FLOAT64_C(   464.21), EASYSIMD_FLOAT64_C(  -220.41) },
      { EASYSIMD_FLOAT64_C(  -868.81), EASYSIMD_FLOAT64_C(  -682.45), EASYSIMD_FLOAT64_C(   376.85), EASYSIMD_FLOAT64_C(   374.58) },
      { EASYSIMD_FLOAT64_C(   868.81), EASYSIMD_FLOAT64_C(   682.45), EASYSIMD_FLOAT64_C(   376.85), EASYSIMD_FLOAT64_C(  -220.41) } }
  };

  easysimd__m256d a, b, r;

  a = easysimd_mm256_loadu_pd(test_vec[0].a);
  b = easysimd_mm256_loadu_pd(test_vec[0].b);
  r = easysimd_mm256_range_pd(a, b, 11);
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[1].a);
  b = easysimd_mm256_loadu_pd(test_vec[1].b);
  r = easysimd_mm256_range_pd(a, b, 4);
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[2].a);
  b = easysimd_mm256_loadu_pd(test_vec[2].b);
  r = easysimd_mm256_range_pd(a, b, 1);
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[3].a);
  b = easysimd_mm256_loadu_pd(test_vec[3].b);
  r = easysimd_mm256_range_pd(a, b, 2);
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[4].a);
  b = easysimd_mm256_loadu_pd(test_vec[4].b);
  r = easysimd_mm256_range_pd(a, b, 3);
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[5].a);
  b = easysimd_mm256_loadu_pd(test_vec[5].b);
  r = easysimd_mm256_range_pd(a, b, 9);
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[6].a);
  b = easysimd_mm256_loadu_pd(test_vec[6].b);
  r = easysimd_mm256_range_pd(a, b, 5);
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm256_loadu_pd(test_vec[7].a);
  b = easysimd_mm256_loadu_pd(test_vec[7].b);
  r = easysimd_mm256_range_pd(a, b, 0);
  easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);

  easysimd__m256d a, b, r;

  a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm256_range_pd(a, b, 11);

  easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm256_range_pd(a, b, 4);

  easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm256_range_pd(a, b, 1);

  easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm256_range_pd(a, b, 2);

  easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm256_range_pd(a, b, 3);

  easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm256_range_pd(a, b, 9);

  easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm256_range_pd(a, b, 5);

  easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm256_range_pd(a, b, 0);

  easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  return 1;
#endif
}

static int
test_easysimd_mm256_mask_range_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  easysimd__m256d src, a, b, e, r;

  src = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   721.40), EASYSIMD_FLOAT64_C(   196.40), EASYSIMD_FLOAT64_C(   859.32), EASYSIMD_FLOAT64_C(  -787.01));
  a = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -160.67), EASYSIMD_FLOAT64_C(   330.35), EASYSIMD_FLOAT64_C(  -715.81), EASYSIMD_FLOAT64_C(  -506.47));
  b = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   913.20), EASYSIMD_FLOAT64_C(   678.56), EASYSIMD_FLOAT64_C(  -770.67), EASYSIMD_FLOAT64_C(  -291.88));
  e = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -913.20), EASYSIMD_FLOAT64_C(   196.40), EASYSIMD_FLOAT64_C(   859.32), EASYSIMD_FLOAT64_C(  -506.47));
  r = easysimd_mm256_mask_range_pd(src, UINT8_C(249), a, b, INT32_C(           3));
  easysimd_test_x86_assert_equal_f64x4(r, e, 1);

  src = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   507.48), EASYSIMD_FLOAT64_C(   -81.12), EASYSIMD_FLOAT64_C(   -74.85), EASYSIMD_FLOAT64_C(   315.28));
  a = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   304.80), EASYSIMD_FLOAT64_C(   583.61), EASYSIMD_FLOAT64_C(   197.96), EASYSIMD_FLOAT64_C(    30.92));
  b = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -859.49), EASYSIMD_FLOAT64_C(  -441.81), EASYSIMD_FLOAT64_C(   147.15), EASYSIMD_FLOAT64_C(  -266.99));
  e = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   507.48), EASYSIMD_FLOAT64_C(   -81.12), EASYSIMD_FLOAT64_C(   147.15), EASYSIMD_FLOAT64_C(   315.28));
  r = easysimd_mm256_mask_range_pd(src, UINT8_C( 66), a, b, INT32_C(           6));
  easysimd_test_x86_assert_equal_f64x4(r, e, 1);

  src = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -455.10), EASYSIMD_FLOAT64_C(   957.60), EASYSIMD_FLOAT64_C(  -664.92), EASYSIMD_FLOAT64_C(  -668.08));
  a = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   647.53), EASYSIMD_FLOAT64_C(   595.89), EASYSIMD_FLOAT64_C(  -733.70), EASYSIMD_FLOAT64_C(   154.00));
  b = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   258.61), EASYSIMD_FLOAT64_C(  -513.15), EASYSIMD_FLOAT64_C(   -73.76), EASYSIMD_FLOAT64_C(  -449.51));
  e = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   647.53), EASYSIMD_FLOAT64_C(   957.60), EASYSIMD_FLOAT64_C(    73.76), EASYSIMD_FLOAT64_C(   154.00));
  r = easysimd_mm256_mask_range_pd(src, UINT8_C(219), a, b, INT32_C(           9));
  easysimd_test_x86_assert_equal_f64x4(r, e, 1);

  src = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   480.69), EASYSIMD_FLOAT64_C(  -302.50), EASYSIMD_FLOAT64_C(   171.81), EASYSIMD_FLOAT64_C(  -834.59));
  a = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -352.70), EASYSIMD_FLOAT64_C(    60.35), EASYSIMD_FLOAT64_C(   -11.83), EASYSIMD_FLOAT64_C(   616.38));
  b = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   -80.86), EASYSIMD_FLOAT64_C(   952.10), EASYSIMD_FLOAT64_C(  -356.05), EASYSIMD_FLOAT64_C(  -813.87));
  e = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   480.69), EASYSIMD_FLOAT64_C(  -302.50), EASYSIMD_FLOAT64_C(  -356.05), EASYSIMD_FLOAT64_C(   813.87));
  r = easysimd_mm256_mask_range_pd(src, UINT8_C( 19), a, b, INT32_C(           0));
  easysimd_test_x86_assert_equal_f64x4(r, e, 1);

  src = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -157.79), EASYSIMD_FLOAT64_C(   397.22), EASYSIMD_FLOAT64_C(    59.65), EASYSIMD_FLOAT64_C(  -489.70));
  a = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -491.19), EASYSIMD_FLOAT64_C(   589.12), EASYSIMD_FLOAT64_C(   387.12), EASYSIMD_FLOAT64_C(   354.82));
  b = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -796.08), EASYSIMD_FLOAT64_C(  -843.66), EASYSIMD_FLOAT64_C(   185.02), EASYSIMD_FLOAT64_C(   653.42));
  e = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -157.79), EASYSIMD_FLOAT64_C(   397.22), EASYSIMD_FLOAT64_C(    59.65), EASYSIMD_FLOAT64_C(  -489.70));
  r = easysimd_mm256_mask_range_pd(src, UINT8_C(128), a, b, INT32_C(          10));
  easysimd_test_x86_assert_equal_f64x4(r, e, 1);

  src = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -191.40), EASYSIMD_FLOAT64_C(   266.82), EASYSIMD_FLOAT64_C(   462.53), EASYSIMD_FLOAT64_C(  -356.81));
  a = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   580.71), EASYSIMD_FLOAT64_C(  -268.71), EASYSIMD_FLOAT64_C(  -710.71), EASYSIMD_FLOAT64_C(   964.32));
  b = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   463.58), EASYSIMD_FLOAT64_C(  -771.99), EASYSIMD_FLOAT64_C(   791.64), EASYSIMD_FLOAT64_C(   277.46));
  e = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   580.71), EASYSIMD_FLOAT64_C(  -268.71), EASYSIMD_FLOAT64_C(   791.64), EASYSIMD_FLOAT64_C(  -356.81));
  r = easysimd_mm256_mask_range_pd(src, UINT8_C(174), a, b, INT32_C(           5));
  easysimd_test_x86_assert_equal_f64x4(r, e, 1);

  src = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -309.59), EASYSIMD_FLOAT64_C(  -773.31), EASYSIMD_FLOAT64_C(  -617.28), EASYSIMD_FLOAT64_C(  -819.89));
  a = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   -21.28), EASYSIMD_FLOAT64_C(  -162.91), EASYSIMD_FLOAT64_C(   532.62), EASYSIMD_FLOAT64_C(   623.91));
  b = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -426.84), EASYSIMD_FLOAT64_C(   487.53), EASYSIMD_FLOAT64_C(  -573.78), EASYSIMD_FLOAT64_C(   -80.27));
  e = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -426.84), EASYSIMD_FLOAT64_C(  -773.31), EASYSIMD_FLOAT64_C(  -573.78), EASYSIMD_FLOAT64_C(   623.91));
  r = easysimd_mm256_mask_range_pd(src, UINT8_C( 75), a, b, INT32_C(           7));
  easysimd_test_x86_assert_equal_f64x4(r, e, 1);

  src = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -712.94), EASYSIMD_FLOAT64_C(   722.49), EASYSIMD_FLOAT64_C(  -222.93), EASYSIMD_FLOAT64_C(   643.87));
  a = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   -46.36), EASYSIMD_FLOAT64_C(  -126.07), EASYSIMD_FLOAT64_C(    95.66), EASYSIMD_FLOAT64_C(   -10.69));
  b = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -337.59), EASYSIMD_FLOAT64_C(  -465.66), EASYSIMD_FLOAT64_C(   605.22), EASYSIMD_FLOAT64_C(   384.95));
  e = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -337.59), EASYSIMD_FLOAT64_C(   722.49), EASYSIMD_FLOAT64_C(  -222.93), EASYSIMD_FLOAT64_C(  -384.95));
  r = easysimd_mm256_mask_range_pd(src, UINT8_C(121), a, b, INT32_C(          15));
  easysimd_test_x86_assert_equal_f64x4(r, e, 1);

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256d src = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    int imm8 = easysimd_test_codegen_rand() & 15;
    easysimd__m256d r;
    EASYSIMD_CONSTIFY_16_(easysimd_mm256_mask_range_pd, r, easysimd_mm256_setzero_pd(), imm8, src, k, a, b);

    easysimd_test_x86_write_f64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_range_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  easysimd__m256d a, b, e, r;

  a = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -491.29), EASYSIMD_FLOAT64_C(   -57.54), EASYSIMD_FLOAT64_C(   832.45), EASYSIMD_FLOAT64_C(  -874.01));
  b = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   683.04), EASYSIMD_FLOAT64_C(   951.08), EASYSIMD_FLOAT64_C(   632.87), EASYSIMD_FLOAT64_C(  -940.87));
  e = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -951.08), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00));
  r = easysimd_mm256_maskz_range_pd(UINT8_C(228), a, b, INT32_C(          13));
  easysimd_test_x86_assert_equal_f64x4(r, e, 1);

  a = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -850.71), EASYSIMD_FLOAT64_C(   214.38), EASYSIMD_FLOAT64_C(  -914.78), EASYSIMD_FLOAT64_C(  -338.24));
  b = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   435.45), EASYSIMD_FLOAT64_C(   793.16), EASYSIMD_FLOAT64_C(  -174.39), EASYSIMD_FLOAT64_C(  -341.62));
  e = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   435.45), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   338.24));
  r = easysimd_mm256_maskz_range_pd(UINT8_C(137), a, b, INT32_C(           9));
  easysimd_test_x86_assert_equal_f64x4(r, e, 1);

  a = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -451.02), EASYSIMD_FLOAT64_C(   175.88), EASYSIMD_FLOAT64_C(   537.41), EASYSIMD_FLOAT64_C(   675.04));
  b = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    25.39), EASYSIMD_FLOAT64_C(  -845.80), EASYSIMD_FLOAT64_C(  -439.17), EASYSIMD_FLOAT64_C(  -508.95));
  e = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   175.88), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   508.95));
  r = easysimd_mm256_maskz_range_pd(UINT8_C( 85), a, b, INT32_C(          10));
  easysimd_test_x86_assert_equal_f64x4(r, e, 1);

  a = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -269.79), EASYSIMD_FLOAT64_C(   383.50), EASYSIMD_FLOAT64_C(   349.23), EASYSIMD_FLOAT64_C(   787.74));
  b = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   809.02), EASYSIMD_FLOAT64_C(  -636.93), EASYSIMD_FLOAT64_C(   442.64), EASYSIMD_FLOAT64_C(   857.94));
  e = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -636.93), EASYSIMD_FLOAT64_C(  -349.23), EASYSIMD_FLOAT64_C(  -787.74));
  r = easysimd_mm256_maskz_range_pd(UINT8_C(119), a, b, INT32_C(          12));
  easysimd_test_x86_assert_equal_f64x4(r, e, 1);

  a = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   811.57), EASYSIMD_FLOAT64_C(   613.78), EASYSIMD_FLOAT64_C(   787.44), EASYSIMD_FLOAT64_C(  -402.82));
  b = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   729.89), EASYSIMD_FLOAT64_C(  -362.82), EASYSIMD_FLOAT64_C(  -727.85), EASYSIMD_FLOAT64_C(   936.73));
  e = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00));
  r = easysimd_mm256_maskz_range_pd(UINT8_C(192), a, b, INT32_C(           8));
  easysimd_test_x86_assert_equal_f64x4(r, e, 1);

  a = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   -14.01), EASYSIMD_FLOAT64_C(  -277.31), EASYSIMD_FLOAT64_C(   382.64), EASYSIMD_FLOAT64_C(   810.11));
  b = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -914.18), EASYSIMD_FLOAT64_C(   546.83), EASYSIMD_FLOAT64_C(   213.74), EASYSIMD_FLOAT64_C(   931.62));
  e = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00));
  r = easysimd_mm256_maskz_range_pd(UINT8_C(160), a, b, INT32_C(           4));
  easysimd_test_x86_assert_equal_f64x4(r, e, 1);

  a = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(    20.38), EASYSIMD_FLOAT64_C(  -880.70), EASYSIMD_FLOAT64_C(  -973.12), EASYSIMD_FLOAT64_C(   636.88));
  b = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(   120.15), EASYSIMD_FLOAT64_C(  -536.98), EASYSIMD_FLOAT64_C(   977.25), EASYSIMD_FLOAT64_C(  -242.92));
  e = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -880.70), EASYSIMD_FLOAT64_C(  -973.12), EASYSIMD_FLOAT64_C(     0.00));
  r = easysimd_mm256_maskz_range_pd(UINT8_C(102), a, b, INT32_C(          12));
  easysimd_test_x86_assert_equal_f64x4(r, e, 1);

  a = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -737.51), EASYSIMD_FLOAT64_C(   376.14), EASYSIMD_FLOAT64_C(  -616.55), EASYSIMD_FLOAT64_C(  -351.29));
  b = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -167.81), EASYSIMD_FLOAT64_C(  -465.36), EASYSIMD_FLOAT64_C(   312.87), EASYSIMD_FLOAT64_C(  -804.98));
  e = easysimd_mm256_set_pd(EASYSIMD_FLOAT64_C(  -737.51), EASYSIMD_FLOAT64_C(  -465.36), EASYSIMD_FLOAT64_C(  -616.55), EASYSIMD_FLOAT64_C(     0.00));
  r = easysimd_mm256_maskz_range_pd(UINT8_C(190), a, b, INT32_C(          15));
  easysimd_test_x86_assert_equal_f64x4(r, e, 1);

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d b = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    int imm8 = easysimd_test_codegen_rand() & 15;
    easysimd__m256d r;
    EASYSIMD_CONSTIFY_16_(easysimd_mm256_maskz_range_pd, r, easysimd_mm256_setzero_pd(), imm8, k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_range_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[8];
    const easysimd_float64 b[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -538.10), EASYSIMD_FLOAT64_C(  -923.13), EASYSIMD_FLOAT64_C(  -505.34), EASYSIMD_FLOAT64_C(  -152.09),
        EASYSIMD_FLOAT64_C(  -619.63), EASYSIMD_FLOAT64_C(   411.91), EASYSIMD_FLOAT64_C(   522.74), EASYSIMD_FLOAT64_C(   733.25) },
      { EASYSIMD_FLOAT64_C(  -577.37), EASYSIMD_FLOAT64_C(   870.81), EASYSIMD_FLOAT64_C(     7.88), EASYSIMD_FLOAT64_C(   327.24),
        EASYSIMD_FLOAT64_C(   570.57), EASYSIMD_FLOAT64_C(   723.95), EASYSIMD_FLOAT64_C(   400.97), EASYSIMD_FLOAT64_C(   241.38) },
      { EASYSIMD_FLOAT64_C(   577.37), EASYSIMD_FLOAT64_C(   923.13), EASYSIMD_FLOAT64_C(   505.34), EASYSIMD_FLOAT64_C(   327.24),
        EASYSIMD_FLOAT64_C(   619.63), EASYSIMD_FLOAT64_C(   723.95), EASYSIMD_FLOAT64_C(   522.74), EASYSIMD_FLOAT64_C(   733.25) } },
    { { EASYSIMD_FLOAT64_C(  -827.60), EASYSIMD_FLOAT64_C(   481.76), EASYSIMD_FLOAT64_C(  -644.01), EASYSIMD_FLOAT64_C(  -199.97),
        EASYSIMD_FLOAT64_C(  -303.44), EASYSIMD_FLOAT64_C(    19.51), EASYSIMD_FLOAT64_C(  -688.85), EASYSIMD_FLOAT64_C(  -378.39) },
      { EASYSIMD_FLOAT64_C(  -473.60), EASYSIMD_FLOAT64_C(   775.37), EASYSIMD_FLOAT64_C(   401.21), EASYSIMD_FLOAT64_C(  -342.40),
        EASYSIMD_FLOAT64_C(  -907.08), EASYSIMD_FLOAT64_C(  -221.95), EASYSIMD_FLOAT64_C(  -967.82), EASYSIMD_FLOAT64_C(  -445.18) },
      { EASYSIMD_FLOAT64_C(  -827.60), EASYSIMD_FLOAT64_C(   481.76), EASYSIMD_FLOAT64_C(  -644.01), EASYSIMD_FLOAT64_C(  -342.40),
        EASYSIMD_FLOAT64_C(  -907.08), EASYSIMD_FLOAT64_C(  -221.95), EASYSIMD_FLOAT64_C(  -967.82), EASYSIMD_FLOAT64_C(  -445.18) } },
    { { EASYSIMD_FLOAT64_C(  -145.08), EASYSIMD_FLOAT64_C(  -473.16), EASYSIMD_FLOAT64_C(   402.73), EASYSIMD_FLOAT64_C(   235.29),
        EASYSIMD_FLOAT64_C(   938.75), EASYSIMD_FLOAT64_C(   -74.53), EASYSIMD_FLOAT64_C(   -31.46), EASYSIMD_FLOAT64_C(  -638.62) },
      { EASYSIMD_FLOAT64_C(  -203.72), EASYSIMD_FLOAT64_C(   976.42), EASYSIMD_FLOAT64_C(   688.62), EASYSIMD_FLOAT64_C(  -633.15),
        EASYSIMD_FLOAT64_C(   700.37), EASYSIMD_FLOAT64_C(    89.59), EASYSIMD_FLOAT64_C(   608.23), EASYSIMD_FLOAT64_C(   872.77) },
      { EASYSIMD_FLOAT64_C(  -145.08), EASYSIMD_FLOAT64_C(  -976.42), EASYSIMD_FLOAT64_C(   688.62), EASYSIMD_FLOAT64_C(   235.29),
        EASYSIMD_FLOAT64_C(   938.75), EASYSIMD_FLOAT64_C(   -89.59), EASYSIMD_FLOAT64_C(  -608.23), EASYSIMD_FLOAT64_C(  -872.77) } },
    { { EASYSIMD_FLOAT64_C(  -428.65), EASYSIMD_FLOAT64_C(   964.22), EASYSIMD_FLOAT64_C(  -327.20), EASYSIMD_FLOAT64_C(   267.91),
        EASYSIMD_FLOAT64_C(   -16.27), EASYSIMD_FLOAT64_C(   -16.05), EASYSIMD_FLOAT64_C(   889.52), EASYSIMD_FLOAT64_C(   510.13) },
      { EASYSIMD_FLOAT64_C(  -240.68), EASYSIMD_FLOAT64_C(   290.73), EASYSIMD_FLOAT64_C(  -832.27), EASYSIMD_FLOAT64_C(  -147.76),
        EASYSIMD_FLOAT64_C(  -931.22), EASYSIMD_FLOAT64_C(  -800.09), EASYSIMD_FLOAT64_C(   407.06), EASYSIMD_FLOAT64_C(   -76.30) },
      { EASYSIMD_FLOAT64_C(  -240.68), EASYSIMD_FLOAT64_C(   290.73), EASYSIMD_FLOAT64_C(  -327.20), EASYSIMD_FLOAT64_C(   147.76),
        EASYSIMD_FLOAT64_C(   -16.27), EASYSIMD_FLOAT64_C(   -16.05), EASYSIMD_FLOAT64_C(   407.06), EASYSIMD_FLOAT64_C(    76.30) } },
    { { EASYSIMD_FLOAT64_C(  -273.25), EASYSIMD_FLOAT64_C(  -190.21), EASYSIMD_FLOAT64_C(  -841.01), EASYSIMD_FLOAT64_C(  -334.50),
        EASYSIMD_FLOAT64_C(   735.25), EASYSIMD_FLOAT64_C(   127.53), EASYSIMD_FLOAT64_C(    26.88), EASYSIMD_FLOAT64_C(  -468.47) },
      { EASYSIMD_FLOAT64_C(   103.95), EASYSIMD_FLOAT64_C(  -284.49), EASYSIMD_FLOAT64_C(  -101.62), EASYSIMD_FLOAT64_C(  -195.68),
        EASYSIMD_FLOAT64_C(   805.10), EASYSIMD_FLOAT64_C(  -493.39), EASYSIMD_FLOAT64_C(  -322.91), EASYSIMD_FLOAT64_C(  -623.55) },
      { EASYSIMD_FLOAT64_C(  -273.25), EASYSIMD_FLOAT64_C(  -284.49), EASYSIMD_FLOAT64_C(  -841.01), EASYSIMD_FLOAT64_C(  -334.50),
        EASYSIMD_FLOAT64_C(   805.10), EASYSIMD_FLOAT64_C(   493.39), EASYSIMD_FLOAT64_C(   322.91), EASYSIMD_FLOAT64_C(  -623.55) } },
    { { EASYSIMD_FLOAT64_C(  -529.17), EASYSIMD_FLOAT64_C(   349.89), EASYSIMD_FLOAT64_C(   644.36), EASYSIMD_FLOAT64_C(   454.56),
        EASYSIMD_FLOAT64_C(  -666.16), EASYSIMD_FLOAT64_C(   533.89), EASYSIMD_FLOAT64_C(   -35.30), EASYSIMD_FLOAT64_C(    93.16) },
      { EASYSIMD_FLOAT64_C(  -175.39), EASYSIMD_FLOAT64_C(   132.43), EASYSIMD_FLOAT64_C(   945.40), EASYSIMD_FLOAT64_C(  -106.61),
        EASYSIMD_FLOAT64_C(   332.34), EASYSIMD_FLOAT64_C(   352.46), EASYSIMD_FLOAT64_C(   817.09), EASYSIMD_FLOAT64_C(  -940.90) },
      { EASYSIMD_FLOAT64_C(   175.39), EASYSIMD_FLOAT64_C(   349.89), EASYSIMD_FLOAT64_C(   945.40), EASYSIMD_FLOAT64_C(   454.56),
        EASYSIMD_FLOAT64_C(   332.34), EASYSIMD_FLOAT64_C(   533.89), EASYSIMD_FLOAT64_C(   817.09), EASYSIMD_FLOAT64_C(    93.16) } },
    { { EASYSIMD_FLOAT64_C(  -837.75), EASYSIMD_FLOAT64_C(   976.08), EASYSIMD_FLOAT64_C(  -275.40), EASYSIMD_FLOAT64_C(   897.50),
        EASYSIMD_FLOAT64_C(   103.61), EASYSIMD_FLOAT64_C(   751.48), EASYSIMD_FLOAT64_C(  -570.97), EASYSIMD_FLOAT64_C(  -792.44) },
      { EASYSIMD_FLOAT64_C(  -533.01), EASYSIMD_FLOAT64_C(   327.42), EASYSIMD_FLOAT64_C(    11.88), EASYSIMD_FLOAT64_C(  -727.91),
        EASYSIMD_FLOAT64_C(   834.03), EASYSIMD_FLOAT64_C(   688.97), EASYSIMD_FLOAT64_C(  -351.45), EASYSIMD_FLOAT64_C(  -695.14) },
      { EASYSIMD_FLOAT64_C(  -533.01), EASYSIMD_FLOAT64_C(   976.08), EASYSIMD_FLOAT64_C(    11.88), EASYSIMD_FLOAT64_C(   897.50),
        EASYSIMD_FLOAT64_C(   834.03), EASYSIMD_FLOAT64_C(   751.48), EASYSIMD_FLOAT64_C(  -351.45), EASYSIMD_FLOAT64_C(  -695.14) } },
    { { EASYSIMD_FLOAT64_C(    38.86), EASYSIMD_FLOAT64_C(  -707.09), EASYSIMD_FLOAT64_C(   759.43), EASYSIMD_FLOAT64_C(   372.70),
        EASYSIMD_FLOAT64_C(   826.80), EASYSIMD_FLOAT64_C(  -275.88), EASYSIMD_FLOAT64_C(  -534.14), EASYSIMD_FLOAT64_C(  -348.59) },
      { EASYSIMD_FLOAT64_C(   856.55), EASYSIMD_FLOAT64_C(  -588.74), EASYSIMD_FLOAT64_C(   544.80), EASYSIMD_FLOAT64_C(   188.90),
        EASYSIMD_FLOAT64_C(   763.72), EASYSIMD_FLOAT64_C(   361.89), EASYSIMD_FLOAT64_C(   247.99), EASYSIMD_FLOAT64_C(   925.97) },
      { EASYSIMD_FLOAT64_C(    38.86), EASYSIMD_FLOAT64_C(  -707.09), EASYSIMD_FLOAT64_C(   544.80), EASYSIMD_FLOAT64_C(   188.90),
        EASYSIMD_FLOAT64_C(   763.72), EASYSIMD_FLOAT64_C(  -275.88), EASYSIMD_FLOAT64_C(  -534.14), EASYSIMD_FLOAT64_C(  -348.59) } }
  };

  easysimd__m512d a, b, r;

  a = easysimd_mm512_loadu_pd(test_vec[0].a);
  b = easysimd_mm512_loadu_pd(test_vec[0].b);
  r = easysimd_mm512_range_pd(a, b, 11);
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[1].a);
  b = easysimd_mm512_loadu_pd(test_vec[1].b);
  r = easysimd_mm512_range_pd(a, b, 4);
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[2].a);
  b = easysimd_mm512_loadu_pd(test_vec[2].b);
  r = easysimd_mm512_range_pd(a, b, 1);
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[3].a);
  b = easysimd_mm512_loadu_pd(test_vec[3].b);
  r = easysimd_mm512_range_pd(a, b, 2);
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[4].a);
  b = easysimd_mm512_loadu_pd(test_vec[4].b);
  r = easysimd_mm512_range_pd(a, b, 3);
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[5].a);
  b = easysimd_mm512_loadu_pd(test_vec[5].b);
  r = easysimd_mm512_range_pd(a, b, 9);
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[6].a);
  b = easysimd_mm512_loadu_pd(test_vec[6].b);
  r = easysimd_mm512_range_pd(a, b, 5);
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm512_loadu_pd(test_vec[7].a);
  b = easysimd_mm512_loadu_pd(test_vec[7].b);
  r = easysimd_mm512_range_pd(a, b, 0);
  easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);

  easysimd__m512d a, b, r;

  a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm512_range_pd(a, b, 11);

  easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm512_range_pd(a, b, 4);

  easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm512_range_pd(a, b, 1);

  easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm512_range_pd(a, b, 2);

  easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm512_range_pd(a, b, 3);

  easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm512_range_pd(a, b, 9);

  easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm512_range_pd(a, b, 5);

  easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
  r = easysimd_mm512_range_pd(a, b, 0);

  easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
  easysimd_test_x86_write_f64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
  easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);

  return 1;
#endif
}

static int
test_easysimd_mm512_mask_range_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  easysimd__m512d src, a, b, e, r;

  src = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -815.01), EASYSIMD_FLOAT64_C(  -900.08), EASYSIMD_FLOAT64_C(   232.89), EASYSIMD_FLOAT64_C(  -900.14),
                           EASYSIMD_FLOAT64_C(  -826.33), EASYSIMD_FLOAT64_C(  -909.36), EASYSIMD_FLOAT64_C(  -362.80), EASYSIMD_FLOAT64_C(  -326.67));
  a = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   489.93), EASYSIMD_FLOAT64_C(    18.24), EASYSIMD_FLOAT64_C(  -633.02), EASYSIMD_FLOAT64_C(  -517.17),
                         EASYSIMD_FLOAT64_C(   -31.50), EASYSIMD_FLOAT64_C(   855.15), EASYSIMD_FLOAT64_C(   229.18), EASYSIMD_FLOAT64_C(   989.52));
  b = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   533.35), EASYSIMD_FLOAT64_C(   215.69), EASYSIMD_FLOAT64_C(   614.42), EASYSIMD_FLOAT64_C(   761.72),
                         EASYSIMD_FLOAT64_C(  -255.85), EASYSIMD_FLOAT64_C(  -511.69), EASYSIMD_FLOAT64_C(  -939.34), EASYSIMD_FLOAT64_C(  -968.03));
  e = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -815.01), EASYSIMD_FLOAT64_C(   215.69), EASYSIMD_FLOAT64_C(   614.42), EASYSIMD_FLOAT64_C(   761.72),
                         EASYSIMD_FLOAT64_C(    31.50), EASYSIMD_FLOAT64_C(   855.15), EASYSIMD_FLOAT64_C(  -362.80), EASYSIMD_FLOAT64_C(  -326.67));
  r = easysimd_mm512_mask_range_pd(src, UINT8_C(124), a, b, INT32_C(           9));
  easysimd_test_x86_assert_equal_f64x8(r, e, 1);

  src = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -853.79), EASYSIMD_FLOAT64_C(  -717.88), EASYSIMD_FLOAT64_C(    33.69), EASYSIMD_FLOAT64_C(  -944.44),
                           EASYSIMD_FLOAT64_C(   644.92), EASYSIMD_FLOAT64_C(  -639.63), EASYSIMD_FLOAT64_C(   606.84), EASYSIMD_FLOAT64_C(  -541.57));
  a = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(    37.59), EASYSIMD_FLOAT64_C(   796.15), EASYSIMD_FLOAT64_C(   296.80), EASYSIMD_FLOAT64_C(   182.44),
                         EASYSIMD_FLOAT64_C(  -433.03), EASYSIMD_FLOAT64_C(   307.27), EASYSIMD_FLOAT64_C(   379.10), EASYSIMD_FLOAT64_C(  -618.02));
  b = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -742.78), EASYSIMD_FLOAT64_C(   344.19), EASYSIMD_FLOAT64_C(   436.55), EASYSIMD_FLOAT64_C(   768.91),
                         EASYSIMD_FLOAT64_C(   283.53), EASYSIMD_FLOAT64_C(   404.57), EASYSIMD_FLOAT64_C(  -721.02), EASYSIMD_FLOAT64_C(  -734.71));
  e = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -853.79), EASYSIMD_FLOAT64_C(  -344.19), EASYSIMD_FLOAT64_C(  -296.80), EASYSIMD_FLOAT64_C(  -944.44),
                         EASYSIMD_FLOAT64_C(   644.92), EASYSIMD_FLOAT64_C(  -307.27), EASYSIMD_FLOAT64_C(  -379.10), EASYSIMD_FLOAT64_C(  -618.02));
  r = easysimd_mm512_mask_range_pd(src, UINT8_C(103), a, b, INT32_C(          14));
  easysimd_test_x86_assert_equal_f64x8(r, e, 1);

  src = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   953.25), EASYSIMD_FLOAT64_C(  -753.90), EASYSIMD_FLOAT64_C(   854.82), EASYSIMD_FLOAT64_C(   592.88),
                           EASYSIMD_FLOAT64_C(  -360.74), EASYSIMD_FLOAT64_C(   396.39), EASYSIMD_FLOAT64_C(   871.64), EASYSIMD_FLOAT64_C(   105.91));
  a = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   501.57), EASYSIMD_FLOAT64_C(   826.97), EASYSIMD_FLOAT64_C(  -836.16), EASYSIMD_FLOAT64_C(  -805.70),
                         EASYSIMD_FLOAT64_C(  -552.13), EASYSIMD_FLOAT64_C(   781.87), EASYSIMD_FLOAT64_C(   -13.06), EASYSIMD_FLOAT64_C(  -698.34));
  b = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   451.58), EASYSIMD_FLOAT64_C(  -194.06), EASYSIMD_FLOAT64_C(    63.66), EASYSIMD_FLOAT64_C(  -953.00),
                         EASYSIMD_FLOAT64_C(  -473.04), EASYSIMD_FLOAT64_C(  -201.63), EASYSIMD_FLOAT64_C(     9.41), EASYSIMD_FLOAT64_C(  -269.19));
  e = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -451.58), EASYSIMD_FLOAT64_C(  -753.90), EASYSIMD_FLOAT64_C(   854.82), EASYSIMD_FLOAT64_C(   592.88),
                         EASYSIMD_FLOAT64_C(  -360.74), EASYSIMD_FLOAT64_C(   396.39), EASYSIMD_FLOAT64_C(    -9.41), EASYSIMD_FLOAT64_C(  -269.19));
  r = easysimd_mm512_mask_range_pd(src, UINT8_C(131), a, b, INT32_C(          14));
  easysimd_test_x86_assert_equal_f64x8(r, e, 1);

  src = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -534.79), EASYSIMD_FLOAT64_C(  -296.28), EASYSIMD_FLOAT64_C(  -202.71), EASYSIMD_FLOAT64_C(    68.82),
                           EASYSIMD_FLOAT64_C(  -167.92), EASYSIMD_FLOAT64_C(   691.38), EASYSIMD_FLOAT64_C(  -111.88), EASYSIMD_FLOAT64_C(  -425.15));
  a = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   601.63), EASYSIMD_FLOAT64_C(   236.79), EASYSIMD_FLOAT64_C(   984.30), EASYSIMD_FLOAT64_C(   819.77),
                         EASYSIMD_FLOAT64_C(  -750.15), EASYSIMD_FLOAT64_C(   682.64), EASYSIMD_FLOAT64_C(  -679.97), EASYSIMD_FLOAT64_C(  -703.40));
  b = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   731.03), EASYSIMD_FLOAT64_C(   268.54), EASYSIMD_FLOAT64_C(  -503.71), EASYSIMD_FLOAT64_C(   -67.34),
                         EASYSIMD_FLOAT64_C(  -740.87), EASYSIMD_FLOAT64_C(   765.48), EASYSIMD_FLOAT64_C(   431.09), EASYSIMD_FLOAT64_C(  -567.84));
  e = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   731.03), EASYSIMD_FLOAT64_C(  -296.28), EASYSIMD_FLOAT64_C(  -202.71), EASYSIMD_FLOAT64_C(    68.82),
                         EASYSIMD_FLOAT64_C(  -167.92), EASYSIMD_FLOAT64_C(   765.48), EASYSIMD_FLOAT64_C(  -431.09), EASYSIMD_FLOAT64_C(  -567.84));
  r = easysimd_mm512_mask_range_pd(src, UINT8_C(135), a, b, INT32_C(           1));
  easysimd_test_x86_assert_equal_f64x8(r, e, 1);

  src = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -166.74), EASYSIMD_FLOAT64_C(   655.25), EASYSIMD_FLOAT64_C(  -595.95), EASYSIMD_FLOAT64_C(   141.88),
                           EASYSIMD_FLOAT64_C(  -232.88), EASYSIMD_FLOAT64_C(   829.19), EASYSIMD_FLOAT64_C(  -205.31), EASYSIMD_FLOAT64_C(   315.54));
  a = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -250.27), EASYSIMD_FLOAT64_C(   509.30), EASYSIMD_FLOAT64_C(  -763.56), EASYSIMD_FLOAT64_C(    67.09),
                         EASYSIMD_FLOAT64_C(   189.28), EASYSIMD_FLOAT64_C(   939.84), EASYSIMD_FLOAT64_C(   630.55), EASYSIMD_FLOAT64_C(  -275.93));
  b = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -303.82), EASYSIMD_FLOAT64_C(   154.17), EASYSIMD_FLOAT64_C(   166.19), EASYSIMD_FLOAT64_C(   -69.30),
                         EASYSIMD_FLOAT64_C(   723.08), EASYSIMD_FLOAT64_C(  -265.98), EASYSIMD_FLOAT64_C(   329.07), EASYSIMD_FLOAT64_C(  -513.71));
  e = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -250.27), EASYSIMD_FLOAT64_C(  -509.30), EASYSIMD_FLOAT64_C(  -595.95), EASYSIMD_FLOAT64_C(   -67.09),
                         EASYSIMD_FLOAT64_C(  -232.88), EASYSIMD_FLOAT64_C(  -939.84), EASYSIMD_FLOAT64_C(  -205.31), EASYSIMD_FLOAT64_C(  -275.93));
  r = easysimd_mm512_mask_range_pd(src, UINT8_C(213), a, b, INT32_C(          13));
  easysimd_test_x86_assert_equal_f64x8(r, e, 1);

  src = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -955.09), EASYSIMD_FLOAT64_C(  -387.44), EASYSIMD_FLOAT64_C(  -990.59), EASYSIMD_FLOAT64_C(  -784.28),
                           EASYSIMD_FLOAT64_C(   817.87), EASYSIMD_FLOAT64_C(  -306.14), EASYSIMD_FLOAT64_C(   192.47), EASYSIMD_FLOAT64_C(  -913.16));
  a = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   624.92), EASYSIMD_FLOAT64_C(  -781.74), EASYSIMD_FLOAT64_C(   155.84), EASYSIMD_FLOAT64_C(   685.08),
                         EASYSIMD_FLOAT64_C(  -412.29), EASYSIMD_FLOAT64_C(  -568.22), EASYSIMD_FLOAT64_C(  -551.04), EASYSIMD_FLOAT64_C(   754.45));
  b = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -230.90), EASYSIMD_FLOAT64_C(   183.49), EASYSIMD_FLOAT64_C(  -652.34), EASYSIMD_FLOAT64_C(  -964.92),
                         EASYSIMD_FLOAT64_C(   854.42), EASYSIMD_FLOAT64_C(   861.36), EASYSIMD_FLOAT64_C(   285.35), EASYSIMD_FLOAT64_C(  -654.88));
  e = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -955.09), EASYSIMD_FLOAT64_C(  -387.44), EASYSIMD_FLOAT64_C(   652.34), EASYSIMD_FLOAT64_C(  -784.28),
                         EASYSIMD_FLOAT64_C(   817.87), EASYSIMD_FLOAT64_C(   861.36), EASYSIMD_FLOAT64_C(   551.04), EASYSIMD_FLOAT64_C(  -913.16));
  r = easysimd_mm512_mask_range_pd(src, UINT8_C( 38), a, b, INT32_C(          11));
  easysimd_test_x86_assert_equal_f64x8(r, e, 1);

  src = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -945.53), EASYSIMD_FLOAT64_C(  -997.15), EASYSIMD_FLOAT64_C(   311.75), EASYSIMD_FLOAT64_C(   360.61),
                           EASYSIMD_FLOAT64_C(  -189.63), EASYSIMD_FLOAT64_C(   224.91), EASYSIMD_FLOAT64_C(   935.29), EASYSIMD_FLOAT64_C(  -885.81));
  a = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   272.18), EASYSIMD_FLOAT64_C(  -287.57), EASYSIMD_FLOAT64_C(   496.63), EASYSIMD_FLOAT64_C(  -159.59),
                         EASYSIMD_FLOAT64_C(  -736.52), EASYSIMD_FLOAT64_C(   742.18), EASYSIMD_FLOAT64_C(  -936.12), EASYSIMD_FLOAT64_C(  -781.43));
  b = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   883.79), EASYSIMD_FLOAT64_C(   587.96), EASYSIMD_FLOAT64_C(  -226.86), EASYSIMD_FLOAT64_C(  -977.57),
                         EASYSIMD_FLOAT64_C(  -697.39), EASYSIMD_FLOAT64_C(  -571.98), EASYSIMD_FLOAT64_C(  -602.49), EASYSIMD_FLOAT64_C(  -915.66));
  e = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   883.79), EASYSIMD_FLOAT64_C(  -997.15), EASYSIMD_FLOAT64_C(   496.63), EASYSIMD_FLOAT64_C(   360.61),
                         EASYSIMD_FLOAT64_C(   736.52), EASYSIMD_FLOAT64_C(   742.18), EASYSIMD_FLOAT64_C(   935.29), EASYSIMD_FLOAT64_C(   915.66));
  r = easysimd_mm512_mask_range_pd(src, UINT8_C(173), a, b, INT32_C(          11));
  easysimd_test_x86_assert_equal_f64x8(r, e, 1);

  src = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   527.10), EASYSIMD_FLOAT64_C(  -672.57), EASYSIMD_FLOAT64_C(   925.24), EASYSIMD_FLOAT64_C(  -697.81),
                           EASYSIMD_FLOAT64_C(  -607.86), EASYSIMD_FLOAT64_C(   811.05), EASYSIMD_FLOAT64_C(  -768.55), EASYSIMD_FLOAT64_C(   623.04));
  a = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   710.66), EASYSIMD_FLOAT64_C(   806.39), EASYSIMD_FLOAT64_C(   -42.97), EASYSIMD_FLOAT64_C(   968.47),
                         EASYSIMD_FLOAT64_C(   742.51), EASYSIMD_FLOAT64_C(  -261.54), EASYSIMD_FLOAT64_C(  -161.15), EASYSIMD_FLOAT64_C(   688.04));
  b = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -653.00), EASYSIMD_FLOAT64_C(  -669.56), EASYSIMD_FLOAT64_C(   291.64), EASYSIMD_FLOAT64_C(   918.98),
                         EASYSIMD_FLOAT64_C(   932.93), EASYSIMD_FLOAT64_C(   207.29), EASYSIMD_FLOAT64_C(  -353.20), EASYSIMD_FLOAT64_C(   220.50));
  e = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   710.66), EASYSIMD_FLOAT64_C(   806.39), EASYSIMD_FLOAT64_C(   291.64), EASYSIMD_FLOAT64_C(   968.47),
                         EASYSIMD_FLOAT64_C(  -607.86), EASYSIMD_FLOAT64_C(   811.05), EASYSIMD_FLOAT64_C(   161.15), EASYSIMD_FLOAT64_C(   623.04));
  r = easysimd_mm512_mask_range_pd(src, UINT8_C(242), a, b, INT32_C(           9));
  easysimd_test_x86_assert_equal_f64x8(r, e, 1);

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512d src = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    int imm8 = easysimd_test_codegen_rand() & 15;
    easysimd__m512d r;
    EASYSIMD_CONSTIFY_16_(easysimd_mm512_mask_range_pd, r, easysimd_mm512_setzero_pd(), imm8, src, k, a, b);

    easysimd_test_x86_write_f64x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_range_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  easysimd__m512d a, b, e, r;

  a = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   197.38), EASYSIMD_FLOAT64_C(   558.75), EASYSIMD_FLOAT64_C(  -531.88), EASYSIMD_FLOAT64_C(  -194.76),
                         EASYSIMD_FLOAT64_C(   747.70), EASYSIMD_FLOAT64_C(  -763.33), EASYSIMD_FLOAT64_C(   182.20), EASYSIMD_FLOAT64_C(   120.14));
  b = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   -41.93), EASYSIMD_FLOAT64_C(   136.26), EASYSIMD_FLOAT64_C(   212.85), EASYSIMD_FLOAT64_C(  -780.39),
                         EASYSIMD_FLOAT64_C(  -702.59), EASYSIMD_FLOAT64_C(   524.81), EASYSIMD_FLOAT64_C(   483.99), EASYSIMD_FLOAT64_C(  -229.69));
  e = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   -41.93), EASYSIMD_FLOAT64_C(   136.26), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),
                         EASYSIMD_FLOAT64_C(  -702.59), EASYSIMD_FLOAT64_C(   524.81), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00));
  r = easysimd_mm512_maskz_range_pd(UINT8_C(204), a, b, INT32_C(           6));
  easysimd_test_x86_assert_equal_f64x8(r, e, 1);

  a = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   327.53), EASYSIMD_FLOAT64_C(    68.54), EASYSIMD_FLOAT64_C(  -977.31), EASYSIMD_FLOAT64_C(   408.55),
                         EASYSIMD_FLOAT64_C(   135.60), EASYSIMD_FLOAT64_C(  -184.60), EASYSIMD_FLOAT64_C(  -238.25), EASYSIMD_FLOAT64_C(   915.10));
  b = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   988.52), EASYSIMD_FLOAT64_C(  -909.23), EASYSIMD_FLOAT64_C(  -205.33), EASYSIMD_FLOAT64_C(   751.85),
                         EASYSIMD_FLOAT64_C(   -91.43), EASYSIMD_FLOAT64_C(   674.53), EASYSIMD_FLOAT64_C(   398.98), EASYSIMD_FLOAT64_C(   314.32));
  e = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    68.54), EASYSIMD_FLOAT64_C(  -205.33), EASYSIMD_FLOAT64_C(   408.55),
                         EASYSIMD_FLOAT64_C(   -91.43), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -238.25), EASYSIMD_FLOAT64_C(     0.00));
  r = easysimd_mm512_maskz_range_pd(UINT8_C(122), a, b, INT32_C(           6));
  easysimd_test_x86_assert_equal_f64x8(r, e, 1);

  a = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   804.72), EASYSIMD_FLOAT64_C(   524.36), EASYSIMD_FLOAT64_C(   618.20), EASYSIMD_FLOAT64_C(   585.11),
                         EASYSIMD_FLOAT64_C(   226.95), EASYSIMD_FLOAT64_C(  -906.62), EASYSIMD_FLOAT64_C(  -898.88), EASYSIMD_FLOAT64_C(  -543.36));
  b = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -419.24), EASYSIMD_FLOAT64_C(  -451.83), EASYSIMD_FLOAT64_C(  -322.11), EASYSIMD_FLOAT64_C(   765.36),
                         EASYSIMD_FLOAT64_C(   786.41), EASYSIMD_FLOAT64_C(  -237.21), EASYSIMD_FLOAT64_C(  -339.38), EASYSIMD_FLOAT64_C(  -168.95));
  e = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   322.11), EASYSIMD_FLOAT64_C(     0.00),
                         EASYSIMD_FLOAT64_C(   226.95), EASYSIMD_FLOAT64_C(  -906.62), EASYSIMD_FLOAT64_C(  -898.88), EASYSIMD_FLOAT64_C(  -543.36));
  r = easysimd_mm512_maskz_range_pd(UINT8_C( 47), a, b, INT32_C(           0));
  easysimd_test_x86_assert_equal_f64x8(r, e, 1);

  a = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -967.14), EASYSIMD_FLOAT64_C(  -173.66), EASYSIMD_FLOAT64_C(   -41.23), EASYSIMD_FLOAT64_C(  -718.99),
                         EASYSIMD_FLOAT64_C(   917.77), EASYSIMD_FLOAT64_C(   284.24), EASYSIMD_FLOAT64_C(  -117.97), EASYSIMD_FLOAT64_C(  -396.55));
  b = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   906.49), EASYSIMD_FLOAT64_C(  -603.07), EASYSIMD_FLOAT64_C(  -521.98), EASYSIMD_FLOAT64_C(   813.11),
                         EASYSIMD_FLOAT64_C(  -704.19), EASYSIMD_FLOAT64_C(  -978.62), EASYSIMD_FLOAT64_C(   -82.90), EASYSIMD_FLOAT64_C(   753.44));
  e = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -967.14), EASYSIMD_FLOAT64_C(  -603.07), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00),
                         EASYSIMD_FLOAT64_C(   704.19), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -117.97), EASYSIMD_FLOAT64_C(  -396.55));
  r = easysimd_mm512_maskz_range_pd(UINT8_C(203), a, b, INT32_C(           0));
  easysimd_test_x86_assert_equal_f64x8(r, e, 1);

  a = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   655.32), EASYSIMD_FLOAT64_C(  -857.85), EASYSIMD_FLOAT64_C(  -450.44), EASYSIMD_FLOAT64_C(   889.95),
                         EASYSIMD_FLOAT64_C(  -644.26), EASYSIMD_FLOAT64_C(   786.76), EASYSIMD_FLOAT64_C(   229.33), EASYSIMD_FLOAT64_C(   524.69));
  b = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   931.28), EASYSIMD_FLOAT64_C(   922.98), EASYSIMD_FLOAT64_C(  -160.48), EASYSIMD_FLOAT64_C(  -352.96),
                         EASYSIMD_FLOAT64_C(    40.94), EASYSIMD_FLOAT64_C(  -763.93), EASYSIMD_FLOAT64_C(  -309.68), EASYSIMD_FLOAT64_C(   227.45));
  e = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -450.44), EASYSIMD_FLOAT64_C(     0.00),
                         EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -786.76), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -524.69));
  r = easysimd_mm512_maskz_range_pd(UINT8_C( 37), a, b, INT32_C(          15));
  easysimd_test_x86_assert_equal_f64x8(r, e, 1);

  a = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -686.16), EASYSIMD_FLOAT64_C(   -60.70), EASYSIMD_FLOAT64_C(  -741.77), EASYSIMD_FLOAT64_C(  -499.27),
                         EASYSIMD_FLOAT64_C(  -356.51), EASYSIMD_FLOAT64_C(  -763.15), EASYSIMD_FLOAT64_C(   583.63), EASYSIMD_FLOAT64_C(  -109.95));
  b = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   105.03), EASYSIMD_FLOAT64_C(   670.56), EASYSIMD_FLOAT64_C(   745.02), EASYSIMD_FLOAT64_C(   318.27),
                         EASYSIMD_FLOAT64_C(  -558.77), EASYSIMD_FLOAT64_C(  -779.67), EASYSIMD_FLOAT64_C(   336.23), EASYSIMD_FLOAT64_C(  -263.74));
  e = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -318.27),
                         EASYSIMD_FLOAT64_C(  -356.51), EASYSIMD_FLOAT64_C(  -763.15), EASYSIMD_FLOAT64_C(   336.23), EASYSIMD_FLOAT64_C(     0.00));
  r = easysimd_mm512_maskz_range_pd(UINT8_C( 30), a, b, INT32_C(           2));
  easysimd_test_x86_assert_equal_f64x8(r, e, 1);

  a = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   580.27), EASYSIMD_FLOAT64_C(   922.98), EASYSIMD_FLOAT64_C(   451.90), EASYSIMD_FLOAT64_C(   -66.77),
                         EASYSIMD_FLOAT64_C(  -117.96), EASYSIMD_FLOAT64_C(   215.82), EASYSIMD_FLOAT64_C(  -757.09), EASYSIMD_FLOAT64_C(   654.59));
  b = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -713.21), EASYSIMD_FLOAT64_C(  -367.66), EASYSIMD_FLOAT64_C(  -598.39), EASYSIMD_FLOAT64_C(  -950.06),
                         EASYSIMD_FLOAT64_C(    48.71), EASYSIMD_FLOAT64_C(   511.55), EASYSIMD_FLOAT64_C(   845.96), EASYSIMD_FLOAT64_C(  -708.58));
  e = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(   580.27), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   451.90), EASYSIMD_FLOAT64_C(     0.00),
                         EASYSIMD_FLOAT64_C(   -48.71), EASYSIMD_FLOAT64_C(   511.55), EASYSIMD_FLOAT64_C(  -845.96), EASYSIMD_FLOAT64_C(     0.00));
  r = easysimd_mm512_maskz_range_pd(UINT8_C(174), a, b, INT32_C(           1));
  easysimd_test_x86_assert_equal_f64x8(r, e, 1);

  a = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -361.11), EASYSIMD_FLOAT64_C(   722.51), EASYSIMD_FLOAT64_C(   667.26), EASYSIMD_FLOAT64_C(   320.63),
                         EASYSIMD_FLOAT64_C(   281.28), EASYSIMD_FLOAT64_C(   446.92), EASYSIMD_FLOAT64_C(   984.40), EASYSIMD_FLOAT64_C(  -454.98));
  b = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(  -830.60), EASYSIMD_FLOAT64_C(   755.95), EASYSIMD_FLOAT64_C(   398.51), EASYSIMD_FLOAT64_C(   -46.43),
                         EASYSIMD_FLOAT64_C(   513.04), EASYSIMD_FLOAT64_C(   743.92), EASYSIMD_FLOAT64_C(   393.06), EASYSIMD_FLOAT64_C(   412.28));
  e = easysimd_mm512_set_pd(EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   320.63),
                         EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   454.98));
  r = easysimd_mm512_maskz_range_pd(UINT8_C( 17), a, b, INT32_C(          11));
  easysimd_test_x86_assert_equal_f64x8(r, e, 1);

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d b = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    int imm8 = easysimd_test_codegen_rand() & 15;
    easysimd__m512d r;
    EASYSIMD_CONSTIFY_16_(easysimd_mm512_maskz_range_pd, r, easysimd_mm512_setzero_pd(), imm8, k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_x_mm_range_ss (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const int imm8;
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   947.75), EASYSIMD_FLOAT32_C(   -97.20), EASYSIMD_FLOAT32_C(  -691.97), EASYSIMD_FLOAT32_C(   313.76) },
      { EASYSIMD_FLOAT32_C(   791.74), EASYSIMD_FLOAT32_C(  -857.28), EASYSIMD_FLOAT32_C(   290.69), EASYSIMD_FLOAT32_C(   907.53) },
       INT32_C(           6),
      { EASYSIMD_FLOAT32_C(   791.74), EASYSIMD_FLOAT32_C(   -97.20), EASYSIMD_FLOAT32_C(  -691.97), EASYSIMD_FLOAT32_C(   313.76) } },
    { { EASYSIMD_FLOAT32_C(  -548.75), EASYSIMD_FLOAT32_C(   688.48), EASYSIMD_FLOAT32_C(   890.89), EASYSIMD_FLOAT32_C(  -755.30) },
      { EASYSIMD_FLOAT32_C(   429.14), EASYSIMD_FLOAT32_C(   889.39), EASYSIMD_FLOAT32_C(  -136.81), EASYSIMD_FLOAT32_C(  -718.79) },
       INT32_C(           3),
      { EASYSIMD_FLOAT32_C(  -548.75), EASYSIMD_FLOAT32_C(   688.48), EASYSIMD_FLOAT32_C(   890.89), EASYSIMD_FLOAT32_C(  -755.30) } },
    { { EASYSIMD_FLOAT32_C(   -45.09), EASYSIMD_FLOAT32_C(   230.04), EASYSIMD_FLOAT32_C(   811.99), EASYSIMD_FLOAT32_C(  -653.58) },
      { EASYSIMD_FLOAT32_C(  -585.24), EASYSIMD_FLOAT32_C(  -266.32), EASYSIMD_FLOAT32_C(  -546.18), EASYSIMD_FLOAT32_C(   265.52) },
       INT32_C(           8),
      { EASYSIMD_FLOAT32_C(   585.24), EASYSIMD_FLOAT32_C(   230.04), EASYSIMD_FLOAT32_C(   811.99), EASYSIMD_FLOAT32_C(  -653.58) } },
    { { EASYSIMD_FLOAT32_C(   324.35), EASYSIMD_FLOAT32_C(   785.92), EASYSIMD_FLOAT32_C(   409.14), EASYSIMD_FLOAT32_C(  -154.92) },
      { EASYSIMD_FLOAT32_C(   733.67), EASYSIMD_FLOAT32_C(  -688.06), EASYSIMD_FLOAT32_C(   153.11), EASYSIMD_FLOAT32_C(    47.43) },
       INT32_C(          14),
      { EASYSIMD_FLOAT32_C(  -324.35), EASYSIMD_FLOAT32_C(   785.92), EASYSIMD_FLOAT32_C(   409.14), EASYSIMD_FLOAT32_C(  -154.92) } },
    { { EASYSIMD_FLOAT32_C(   295.83), EASYSIMD_FLOAT32_C(  -661.88), EASYSIMD_FLOAT32_C(  -988.80), EASYSIMD_FLOAT32_C(  -430.24) },
      { EASYSIMD_FLOAT32_C(  -210.63), EASYSIMD_FLOAT32_C(   699.69), EASYSIMD_FLOAT32_C(  -539.35), EASYSIMD_FLOAT32_C(    34.06) },
       INT32_C(          14),
      { EASYSIMD_FLOAT32_C(  -210.63), EASYSIMD_FLOAT32_C(  -661.88), EASYSIMD_FLOAT32_C(  -988.80), EASYSIMD_FLOAT32_C(  -430.24) } },
    { { EASYSIMD_FLOAT32_C(  -649.97), EASYSIMD_FLOAT32_C(   897.25), EASYSIMD_FLOAT32_C(   410.05), EASYSIMD_FLOAT32_C(  -334.38) },
      { EASYSIMD_FLOAT32_C(  -147.84), EASYSIMD_FLOAT32_C(  -359.91), EASYSIMD_FLOAT32_C(  -522.39), EASYSIMD_FLOAT32_C(   198.57) },
       INT32_C(           5),
      { EASYSIMD_FLOAT32_C(  -147.84), EASYSIMD_FLOAT32_C(   897.25), EASYSIMD_FLOAT32_C(   410.05), EASYSIMD_FLOAT32_C(  -334.38) } },
    { { EASYSIMD_FLOAT32_C(   211.29), EASYSIMD_FLOAT32_C(   652.40), EASYSIMD_FLOAT32_C(  -679.63), EASYSIMD_FLOAT32_C(  -382.50) },
      { EASYSIMD_FLOAT32_C(   -23.25), EASYSIMD_FLOAT32_C(  -893.71), EASYSIMD_FLOAT32_C(  -973.36), EASYSIMD_FLOAT32_C(   821.82) },
       INT32_C(          12),
      { EASYSIMD_FLOAT32_C(   -23.25), EASYSIMD_FLOAT32_C(   652.40), EASYSIMD_FLOAT32_C(  -679.63), EASYSIMD_FLOAT32_C(  -382.50) } },
    { { EASYSIMD_FLOAT32_C(  -661.42), EASYSIMD_FLOAT32_C(   -25.07), EASYSIMD_FLOAT32_C(  -112.61), EASYSIMD_FLOAT32_C(  -557.75) },
      { EASYSIMD_FLOAT32_C(  -729.24), EASYSIMD_FLOAT32_C(   225.51), EASYSIMD_FLOAT32_C(  -546.54), EASYSIMD_FLOAT32_C(  -159.48) },
       INT32_C(           8),
      { EASYSIMD_FLOAT32_C(   729.24), EASYSIMD_FLOAT32_C(   -25.07), EASYSIMD_FLOAT32_C(  -112.61), EASYSIMD_FLOAT32_C(  -557.75) } },
  };

  easysimd__m128 a, b, r;

  a = easysimd_mm_loadu_ps(test_vec[0].a);
  b = easysimd_mm_loadu_ps(test_vec[0].b);
  r = easysimd_x_mm_range_ss(a, b, INT32_C(           6));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[1].a);
  b = easysimd_mm_loadu_ps(test_vec[1].b);
  r = easysimd_x_mm_range_ss(a, b, INT32_C(           3));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[2].a);
  b = easysimd_mm_loadu_ps(test_vec[2].b);
  r = easysimd_x_mm_range_ss(a, b, INT32_C(           8));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[3].a);
  b = easysimd_mm_loadu_ps(test_vec[3].b);
  r = easysimd_x_mm_range_ss(a, b, INT32_C(          14));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[4].a);
  b = easysimd_mm_loadu_ps(test_vec[4].b);
  r = easysimd_x_mm_range_ss(a, b, INT32_C(          14));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[5].a);
  b = easysimd_mm_loadu_ps(test_vec[5].b);
  r = easysimd_x_mm_range_ss(a, b, INT32_C(           5));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[6].a);
  b = easysimd_mm_loadu_ps(test_vec[6].b);
  r = easysimd_x_mm_range_ss(a, b, INT32_C(           12));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[7].a);
  b = easysimd_mm_loadu_ps(test_vec[7].b);
  r = easysimd_x_mm_range_ss(a, b, INT32_C(           8));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    int imm8 = easysimd_test_codegen_rand() & 15;
    easysimd__m128 r;
    EASYSIMD_CONSTIFY_16_(easysimd_x_mm_range_ss, r, easysimd_mm_setzero_ps(), imm8, a, b);

    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_range_ss (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[4];
    const easysimd__mmask8 k;
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const int imm8;
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   -57.69), EASYSIMD_FLOAT32_C(   161.65), EASYSIMD_FLOAT32_C(  -852.14), EASYSIMD_FLOAT32_C(   635.17) },
      UINT8_C( 52),
      { EASYSIMD_FLOAT32_C(   468.35), EASYSIMD_FLOAT32_C(   724.81), EASYSIMD_FLOAT32_C(   577.52), EASYSIMD_FLOAT32_C(   574.19) },
      { EASYSIMD_FLOAT32_C(   277.24), EASYSIMD_FLOAT32_C(   787.90), EASYSIMD_FLOAT32_C(   160.06), EASYSIMD_FLOAT32_C(   -11.06) },
       INT32_C(           5),
      { EASYSIMD_FLOAT32_C(   -57.69), EASYSIMD_FLOAT32_C(   724.81), EASYSIMD_FLOAT32_C(   577.52), EASYSIMD_FLOAT32_C(   574.19) } },
    { { EASYSIMD_FLOAT32_C(  -380.86), EASYSIMD_FLOAT32_C(   579.10), EASYSIMD_FLOAT32_C(   673.02), EASYSIMD_FLOAT32_C(   346.64) },
      UINT8_C(213),
      { EASYSIMD_FLOAT32_C(   901.03), EASYSIMD_FLOAT32_C(   930.89), EASYSIMD_FLOAT32_C(  -191.46), EASYSIMD_FLOAT32_C(   441.50) },
      { EASYSIMD_FLOAT32_C(  -285.40), EASYSIMD_FLOAT32_C(   166.01), EASYSIMD_FLOAT32_C(   358.79), EASYSIMD_FLOAT32_C(   892.65) },
       INT32_C(           2),
      { EASYSIMD_FLOAT32_C(   285.40), EASYSIMD_FLOAT32_C(   930.89), EASYSIMD_FLOAT32_C(  -191.46), EASYSIMD_FLOAT32_C(   441.50) } },
    { { EASYSIMD_FLOAT32_C(   514.76), EASYSIMD_FLOAT32_C(  -820.75), EASYSIMD_FLOAT32_C(   636.79), EASYSIMD_FLOAT32_C(  -542.93) },
      UINT8_C( 94),
      { EASYSIMD_FLOAT32_C(   784.64), EASYSIMD_FLOAT32_C(  -907.77), EASYSIMD_FLOAT32_C(  -431.78), EASYSIMD_FLOAT32_C(   252.99) },
      { EASYSIMD_FLOAT32_C(   817.04), EASYSIMD_FLOAT32_C(  -854.26), EASYSIMD_FLOAT32_C(  -172.82), EASYSIMD_FLOAT32_C(    94.28) },
       INT32_C(           8),
      { EASYSIMD_FLOAT32_C(   514.76), EASYSIMD_FLOAT32_C(  -907.77), EASYSIMD_FLOAT32_C(  -431.78), EASYSIMD_FLOAT32_C(   252.99) } },
    { { EASYSIMD_FLOAT32_C(   987.24), EASYSIMD_FLOAT32_C(  -916.78), EASYSIMD_FLOAT32_C(  -756.04), EASYSIMD_FLOAT32_C(  -393.62) },
      UINT8_C( 38),
      { EASYSIMD_FLOAT32_C(   916.98), EASYSIMD_FLOAT32_C(   953.02), EASYSIMD_FLOAT32_C(   305.22), EASYSIMD_FLOAT32_C(   818.01) },
      { EASYSIMD_FLOAT32_C(   883.91), EASYSIMD_FLOAT32_C(  -886.23), EASYSIMD_FLOAT32_C(   259.51), EASYSIMD_FLOAT32_C(  -401.49) },
       INT32_C(           9),
      { EASYSIMD_FLOAT32_C(   987.24), EASYSIMD_FLOAT32_C(   953.02), EASYSIMD_FLOAT32_C(   305.22), EASYSIMD_FLOAT32_C(   818.01) } },
    { { EASYSIMD_FLOAT32_C(  -381.70), EASYSIMD_FLOAT32_C(  -508.84), EASYSIMD_FLOAT32_C(  -780.32), EASYSIMD_FLOAT32_C(  -866.94) },
      UINT8_C( 11),
      { EASYSIMD_FLOAT32_C(   856.47), EASYSIMD_FLOAT32_C(  -409.87), EASYSIMD_FLOAT32_C(  -988.68), EASYSIMD_FLOAT32_C(   641.11) },
      { EASYSIMD_FLOAT32_C(  -317.64), EASYSIMD_FLOAT32_C(  -420.46), EASYSIMD_FLOAT32_C(  -105.90), EASYSIMD_FLOAT32_C(  -500.59) },
       INT32_C(           0),
      { EASYSIMD_FLOAT32_C(   317.64), EASYSIMD_FLOAT32_C(  -409.87), EASYSIMD_FLOAT32_C(  -988.68), EASYSIMD_FLOAT32_C(   641.11) } },
    { { EASYSIMD_FLOAT32_C(   721.27), EASYSIMD_FLOAT32_C(   593.69), EASYSIMD_FLOAT32_C(  -341.09), EASYSIMD_FLOAT32_C(   708.51) },
      UINT8_C(181),
      { EASYSIMD_FLOAT32_C(   -97.13), EASYSIMD_FLOAT32_C(  -685.11), EASYSIMD_FLOAT32_C(   339.23), EASYSIMD_FLOAT32_C(  -180.15) },
      { EASYSIMD_FLOAT32_C(  -732.09), EASYSIMD_FLOAT32_C(  -355.55), EASYSIMD_FLOAT32_C(  -362.14), EASYSIMD_FLOAT32_C(  -848.18) },
       INT32_C(           5),
      { EASYSIMD_FLOAT32_C(   -97.13), EASYSIMD_FLOAT32_C(  -685.11), EASYSIMD_FLOAT32_C(   339.23), EASYSIMD_FLOAT32_C(  -180.15) } },
    { { EASYSIMD_FLOAT32_C(   897.37), EASYSIMD_FLOAT32_C(  -249.67), EASYSIMD_FLOAT32_C(  -962.01), EASYSIMD_FLOAT32_C(  -484.33) },
         UINT8_MAX,
      { EASYSIMD_FLOAT32_C(  -742.32), EASYSIMD_FLOAT32_C(  -351.27), EASYSIMD_FLOAT32_C(   911.91), EASYSIMD_FLOAT32_C(  -885.86) },
      { EASYSIMD_FLOAT32_C(   238.86), EASYSIMD_FLOAT32_C(   923.22), EASYSIMD_FLOAT32_C(   755.25), EASYSIMD_FLOAT32_C(   921.23) },
       INT32_C(           1),
      { EASYSIMD_FLOAT32_C(  -238.86), EASYSIMD_FLOAT32_C(  -351.27), EASYSIMD_FLOAT32_C(   911.91), EASYSIMD_FLOAT32_C(  -885.86) } },
    { { EASYSIMD_FLOAT32_C(  -350.65), EASYSIMD_FLOAT32_C(  -579.36), EASYSIMD_FLOAT32_C(   228.04), EASYSIMD_FLOAT32_C(  -629.38) },
      UINT8_C( 81),
      { EASYSIMD_FLOAT32_C(   886.95), EASYSIMD_FLOAT32_C(  -920.86), EASYSIMD_FLOAT32_C(   691.24), EASYSIMD_FLOAT32_C(  -210.18) },
      { EASYSIMD_FLOAT32_C(  -605.97), EASYSIMD_FLOAT32_C(    30.47), EASYSIMD_FLOAT32_C(   609.67), EASYSIMD_FLOAT32_C(  -338.06) },
       INT32_C(           9),
      { EASYSIMD_FLOAT32_C(   886.95), EASYSIMD_FLOAT32_C(  -920.86), EASYSIMD_FLOAT32_C(   691.24), EASYSIMD_FLOAT32_C(  -210.18) } },
  };

  easysimd__m128 src, a, b, r;

  src = easysimd_mm_loadu_ps(test_vec[0].src);
  a = easysimd_mm_loadu_ps(test_vec[0].a);
  b = easysimd_mm_loadu_ps(test_vec[0].b);
  r = easysimd_mm_mask_range_ss(src, test_vec[0].k, a, b, INT32_C(           5));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[0].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[1].src);
  a = easysimd_mm_loadu_ps(test_vec[1].a);
  b = easysimd_mm_loadu_ps(test_vec[1].b);
  r = easysimd_mm_mask_range_ss(src, test_vec[1].k, a, b, INT32_C(           2));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[1].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[2].src);
  a = easysimd_mm_loadu_ps(test_vec[2].a);
  b = easysimd_mm_loadu_ps(test_vec[2].b);
  r = easysimd_mm_mask_range_ss(src, test_vec[2].k, a, b, INT32_C(           8));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[2].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[3].src);
  a = easysimd_mm_loadu_ps(test_vec[3].a);
  b = easysimd_mm_loadu_ps(test_vec[3].b);
  r = easysimd_mm_mask_range_ss(src, test_vec[3].k, a, b, INT32_C(           9));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[3].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[4].src);
  a = easysimd_mm_loadu_ps(test_vec[4].a);
  b = easysimd_mm_loadu_ps(test_vec[4].b);
  r = easysimd_mm_mask_range_ss(src, test_vec[4].k, a, b, INT32_C(           0));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[4].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[5].src);
  a = easysimd_mm_loadu_ps(test_vec[5].a);
  b = easysimd_mm_loadu_ps(test_vec[5].b);
  r = easysimd_mm_mask_range_ss(src, test_vec[5].k, a, b, INT32_C(           5));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[5].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[6].src);
  a = easysimd_mm_loadu_ps(test_vec[6].a);
  b = easysimd_mm_loadu_ps(test_vec[6].b);
  r = easysimd_mm_mask_range_ss(src, test_vec[6].k, a, b, INT32_C(           1));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[6].r), 1);

  src = easysimd_mm_loadu_ps(test_vec[7].src);
  a = easysimd_mm_loadu_ps(test_vec[7].a);
  b = easysimd_mm_loadu_ps(test_vec[7].b);
  r = easysimd_mm_mask_range_ss(src, test_vec[7].k, a, b, INT32_C(           9));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    int imm8 = easysimd_test_codegen_rand() & 15;
    easysimd__m128 r;
    EASYSIMD_CONSTIFY_16_(easysimd_mm_mask_range_ss, r, easysimd_mm_setzero_ps(), imm8, src, k, a, b);

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
test_easysimd_mm_maskz_range_ss (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float32 a[4];
    const easysimd_float32 b[4];
    const int imm8;
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C( 87),
      { EASYSIMD_FLOAT32_C(  -186.23), EASYSIMD_FLOAT32_C(  -566.86), EASYSIMD_FLOAT32_C(  -855.11), EASYSIMD_FLOAT32_C(   564.10) },
      { EASYSIMD_FLOAT32_C(  -528.86), EASYSIMD_FLOAT32_C(  -339.43), EASYSIMD_FLOAT32_C(  -194.41), EASYSIMD_FLOAT32_C(  -271.19) },
       INT32_C(           0),
      { EASYSIMD_FLOAT32_C(  -528.86), EASYSIMD_FLOAT32_C(  -566.86), EASYSIMD_FLOAT32_C(  -855.11), EASYSIMD_FLOAT32_C(   564.10) } },
    { UINT8_C( 73),
      { EASYSIMD_FLOAT32_C(  -157.04), EASYSIMD_FLOAT32_C(  -451.84), EASYSIMD_FLOAT32_C(  -359.27), EASYSIMD_FLOAT32_C(  -401.79) },
      { EASYSIMD_FLOAT32_C(  -530.61), EASYSIMD_FLOAT32_C(   143.49), EASYSIMD_FLOAT32_C(   247.56), EASYSIMD_FLOAT32_C(  -109.97) },
       INT32_C(           1),
      { EASYSIMD_FLOAT32_C(  -157.04), EASYSIMD_FLOAT32_C(  -451.84), EASYSIMD_FLOAT32_C(  -359.27), EASYSIMD_FLOAT32_C(  -401.79) } },
    { UINT8_C( 66),
      { EASYSIMD_FLOAT32_C(   -95.65), EASYSIMD_FLOAT32_C(  -741.53), EASYSIMD_FLOAT32_C(   697.32), EASYSIMD_FLOAT32_C(  -404.41) },
      { EASYSIMD_FLOAT32_C(    48.29), EASYSIMD_FLOAT32_C(  -908.65), EASYSIMD_FLOAT32_C(   626.06), EASYSIMD_FLOAT32_C(  -342.04) },
       INT32_C(          12),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -741.53), EASYSIMD_FLOAT32_C(   697.32), EASYSIMD_FLOAT32_C(  -404.41) } },
    { UINT8_C( 29),
      { EASYSIMD_FLOAT32_C(   -94.52), EASYSIMD_FLOAT32_C(   567.06), EASYSIMD_FLOAT32_C(   734.13), EASYSIMD_FLOAT32_C(    50.37) },
      { EASYSIMD_FLOAT32_C(   131.16), EASYSIMD_FLOAT32_C(  -794.74), EASYSIMD_FLOAT32_C(   710.94), EASYSIMD_FLOAT32_C(   936.75) },
       INT32_C(          12),
      { EASYSIMD_FLOAT32_C(   -94.52), EASYSIMD_FLOAT32_C(   567.06), EASYSIMD_FLOAT32_C(   734.13), EASYSIMD_FLOAT32_C(    50.37) } },
    { UINT8_C( 54),
      { EASYSIMD_FLOAT32_C(  -345.75), EASYSIMD_FLOAT32_C(   777.03), EASYSIMD_FLOAT32_C(   568.40), EASYSIMD_FLOAT32_C(   294.98) },
      { EASYSIMD_FLOAT32_C(  -624.76), EASYSIMD_FLOAT32_C(  -962.21), EASYSIMD_FLOAT32_C(  -561.53), EASYSIMD_FLOAT32_C(   622.80) },
       INT32_C(           1),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   777.03), EASYSIMD_FLOAT32_C(   568.40), EASYSIMD_FLOAT32_C(   294.98) } },
    { UINT8_C(178),
      { EASYSIMD_FLOAT32_C(   240.99), EASYSIMD_FLOAT32_C(   832.17), EASYSIMD_FLOAT32_C(    68.47), EASYSIMD_FLOAT32_C(   -61.69) },
      { EASYSIMD_FLOAT32_C(  -572.24), EASYSIMD_FLOAT32_C(  -883.24), EASYSIMD_FLOAT32_C(    29.66), EASYSIMD_FLOAT32_C(  -946.18) },
       INT32_C(           3),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   832.17), EASYSIMD_FLOAT32_C(    68.47), EASYSIMD_FLOAT32_C(   -61.69) } },
    { UINT8_C(138),
      { EASYSIMD_FLOAT32_C(   354.81), EASYSIMD_FLOAT32_C(   680.19), EASYSIMD_FLOAT32_C(   350.02), EASYSIMD_FLOAT32_C(    88.93) },
      { EASYSIMD_FLOAT32_C(  -269.44), EASYSIMD_FLOAT32_C(  -518.83), EASYSIMD_FLOAT32_C(   294.20), EASYSIMD_FLOAT32_C(  -558.50) },
       INT32_C(          13),
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   680.19), EASYSIMD_FLOAT32_C(   350.02), EASYSIMD_FLOAT32_C(    88.93) } },
    { UINT8_C(129),
      { EASYSIMD_FLOAT32_C(   461.74), EASYSIMD_FLOAT32_C(    72.18), EASYSIMD_FLOAT32_C(  -994.69), EASYSIMD_FLOAT32_C(    30.14) },
      { EASYSIMD_FLOAT32_C(  -632.84), EASYSIMD_FLOAT32_C(  -619.45), EASYSIMD_FLOAT32_C(    67.93), EASYSIMD_FLOAT32_C(  -194.38) },
       INT32_C(           4),
      { EASYSIMD_FLOAT32_C(  -632.84), EASYSIMD_FLOAT32_C(    72.18), EASYSIMD_FLOAT32_C(  -994.69), EASYSIMD_FLOAT32_C(    30.14) } },
  };

  easysimd__m128 a, b, r;

  a = easysimd_mm_loadu_ps(test_vec[0].a);
  b = easysimd_mm_loadu_ps(test_vec[0].b);
  r = easysimd_mm_maskz_range_ss(test_vec[0].k, a, b, INT32_C(           0));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[0].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[1].a);
  b = easysimd_mm_loadu_ps(test_vec[1].b);
  r = easysimd_mm_maskz_range_ss(test_vec[1].k, a, b, INT32_C(           1));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[1].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[2].a);
  b = easysimd_mm_loadu_ps(test_vec[2].b);
  r = easysimd_mm_maskz_range_ss(test_vec[2].k, a, b, INT32_C(          12));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[2].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[3].a);
  b = easysimd_mm_loadu_ps(test_vec[3].b);
  r = easysimd_mm_maskz_range_ss(test_vec[3].k, a, b, INT32_C(          12));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[3].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[4].a);
  b = easysimd_mm_loadu_ps(test_vec[4].b);
  r = easysimd_mm_maskz_range_ss(test_vec[4].k, a, b, INT32_C(           1));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[4].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[5].a);
  b = easysimd_mm_loadu_ps(test_vec[5].b);
  r = easysimd_mm_maskz_range_ss(test_vec[5].k, a, b, INT32_C(           2));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[5].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[6].a);
  b = easysimd_mm_loadu_ps(test_vec[6].b);
  r = easysimd_mm_maskz_range_ss(test_vec[6].k, a, b, INT32_C(          13));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[6].r), 1);

  a = easysimd_mm_loadu_ps(test_vec[7].a);
  b = easysimd_mm_loadu_ps(test_vec[7].b);
  r = easysimd_mm_maskz_range_ss(test_vec[7].k, a, b, INT32_C(           4));
  easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128 b = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    int imm8 = easysimd_test_codegen_rand() & 15;
    easysimd__m128 r;
    EASYSIMD_CONSTIFY_16_(easysimd_mm_maskz_range_ss, r, easysimd_mm_setzero_ps(), imm8, k, a, b);

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
test_easysimd_x_mm_range_sd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const int imm8;
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   303.71), EASYSIMD_FLOAT64_C(   529.35) },
      { EASYSIMD_FLOAT64_C(   873.73), EASYSIMD_FLOAT64_C(    90.41) },
       INT32_C(           8),
      { EASYSIMD_FLOAT64_C(   303.71), EASYSIMD_FLOAT64_C(   529.35) } },
    { { EASYSIMD_FLOAT64_C(   435.31), EASYSIMD_FLOAT64_C(     7.29) },
      { EASYSIMD_FLOAT64_C(  -660.60), EASYSIMD_FLOAT64_C(   738.01) },
       INT32_C(           0),
      { EASYSIMD_FLOAT64_C(   660.60), EASYSIMD_FLOAT64_C(     7.29) } },
    { { EASYSIMD_FLOAT64_C(  -704.98), EASYSIMD_FLOAT64_C(  -155.73) },
      { EASYSIMD_FLOAT64_C(  -431.88), EASYSIMD_FLOAT64_C(   554.77) },
       INT32_C(           9),
      { EASYSIMD_FLOAT64_C(   431.88), EASYSIMD_FLOAT64_C(  -155.73) } },
    { { EASYSIMD_FLOAT64_C(   586.62), EASYSIMD_FLOAT64_C(  -121.06) },
      { EASYSIMD_FLOAT64_C(   520.15), EASYSIMD_FLOAT64_C(   384.57) },
       INT32_C(           5),
      { EASYSIMD_FLOAT64_C(   586.62), EASYSIMD_FLOAT64_C(  -121.06) } },
    { { EASYSIMD_FLOAT64_C(   -80.52), EASYSIMD_FLOAT64_C(     0.12) },
      { EASYSIMD_FLOAT64_C(  -109.85), EASYSIMD_FLOAT64_C(   256.11) },
       INT32_C(          14),
      { EASYSIMD_FLOAT64_C(   -80.52), EASYSIMD_FLOAT64_C(     0.12) } },
    { { EASYSIMD_FLOAT64_C(  -228.78), EASYSIMD_FLOAT64_C(  -140.83) },
      { EASYSIMD_FLOAT64_C(   360.44), EASYSIMD_FLOAT64_C(  -282.81) },
       INT32_C(           6),
      { EASYSIMD_FLOAT64_C(  -228.78), EASYSIMD_FLOAT64_C(  -140.83) } },
    { { EASYSIMD_FLOAT64_C(  -452.84), EASYSIMD_FLOAT64_C(  -979.10) },
      { EASYSIMD_FLOAT64_C(  -447.11), EASYSIMD_FLOAT64_C(  -579.11) },
       INT32_C(          11),
      { EASYSIMD_FLOAT64_C(   452.84), EASYSIMD_FLOAT64_C(  -979.10) } },
    { { EASYSIMD_FLOAT64_C(   268.05), EASYSIMD_FLOAT64_C(   856.20) },
      { EASYSIMD_FLOAT64_C(  -881.39), EASYSIMD_FLOAT64_C(   607.45) },
       INT32_C(           5),
      { EASYSIMD_FLOAT64_C(   268.05), EASYSIMD_FLOAT64_C(   856.20) } },
  };

  easysimd__m128d a, b, r;

  a = easysimd_mm_loadu_pd(test_vec[0].a);
  b = easysimd_mm_loadu_pd(test_vec[0].b);
  r = easysimd_x_mm_range_sd(a, b, INT32_C(           8));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[1].a);
  b = easysimd_mm_loadu_pd(test_vec[1].b);
  r = easysimd_x_mm_range_sd(a, b, INT32_C(           0));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[2].a);
  b = easysimd_mm_loadu_pd(test_vec[2].b);
  r = easysimd_x_mm_range_sd(a, b, INT32_C(           9));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[3].a);
  b = easysimd_mm_loadu_pd(test_vec[3].b);
  r = easysimd_x_mm_range_sd(a, b, INT32_C(           5));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[4].a);
  b = easysimd_mm_loadu_pd(test_vec[4].b);
  r = easysimd_x_mm_range_sd(a, b, INT32_C(           14));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[5].a);
  b = easysimd_mm_loadu_pd(test_vec[5].b);
  r = easysimd_x_mm_range_sd(a, b, INT32_C(            6));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[6].a);
  b = easysimd_mm_loadu_pd(test_vec[6].b);
  r = easysimd_x_mm_range_sd(a, b, INT32_C(           11));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[7].a);
  b = easysimd_mm_loadu_pd(test_vec[7].b);
  r = easysimd_x_mm_range_sd(a, b, INT32_C(            5));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    int imm8 = easysimd_test_codegen_rand() & 15;
    easysimd__m128d r;
    EASYSIMD_CONSTIFY_16_(easysimd_x_mm_range_sd, r, easysimd_mm_setzero_pd(), imm8, a, b);

    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_range_sd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[2];
    const easysimd__mmask8 k;
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const int imm8;
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   233.87), EASYSIMD_FLOAT64_C(  -459.39) },
      UINT8_C(157),
      { EASYSIMD_FLOAT64_C(    72.78), EASYSIMD_FLOAT64_C(   522.02) },
      { EASYSIMD_FLOAT64_C(  -642.47), EASYSIMD_FLOAT64_C(   396.59) },
       INT32_C(           4),
      { EASYSIMD_FLOAT64_C(  -642.47), EASYSIMD_FLOAT64_C(   522.02) } },
    { { EASYSIMD_FLOAT64_C(  -430.17), EASYSIMD_FLOAT64_C(   351.20) },
      UINT8_C(216),
      { EASYSIMD_FLOAT64_C(  -280.98), EASYSIMD_FLOAT64_C(  -547.40) },
      { EASYSIMD_FLOAT64_C(  -695.26), EASYSIMD_FLOAT64_C(  -362.72) },
       INT32_C(           2),
      { EASYSIMD_FLOAT64_C(  -430.17), EASYSIMD_FLOAT64_C(  -547.40) } },
    { { EASYSIMD_FLOAT64_C(  -391.21), EASYSIMD_FLOAT64_C(  -190.43) },
      UINT8_C(242),
      { EASYSIMD_FLOAT64_C(   422.87), EASYSIMD_FLOAT64_C(    12.55) },
      { EASYSIMD_FLOAT64_C(  -439.83), EASYSIMD_FLOAT64_C(   770.04) },
       INT32_C(          14),
      { EASYSIMD_FLOAT64_C(  -391.21), EASYSIMD_FLOAT64_C(    12.55) } },
    { { EASYSIMD_FLOAT64_C(   363.45), EASYSIMD_FLOAT64_C(  -802.30) },
      UINT8_C(140),
      { EASYSIMD_FLOAT64_C(   744.84), EASYSIMD_FLOAT64_C(   780.99) },
      { EASYSIMD_FLOAT64_C(  -987.02), EASYSIMD_FLOAT64_C(   695.51) },
       INT32_C(          13),
      { EASYSIMD_FLOAT64_C(   363.45), EASYSIMD_FLOAT64_C(   780.99) } },
    { { EASYSIMD_FLOAT64_C(  -446.41), EASYSIMD_FLOAT64_C(  -577.07) },
      UINT8_C(242),
      { EASYSIMD_FLOAT64_C(  -924.38), EASYSIMD_FLOAT64_C(  -219.54) },
      { EASYSIMD_FLOAT64_C(   484.24), EASYSIMD_FLOAT64_C(   -87.33) },
       INT32_C(           5),
      { EASYSIMD_FLOAT64_C(  -446.41), EASYSIMD_FLOAT64_C(  -219.54) } },
    { { EASYSIMD_FLOAT64_C(  -164.56), EASYSIMD_FLOAT64_C(  -940.26) },
      UINT8_C(  0),
      { EASYSIMD_FLOAT64_C(   288.03), EASYSIMD_FLOAT64_C(  -635.52) },
      { EASYSIMD_FLOAT64_C(  -293.41), EASYSIMD_FLOAT64_C(   491.32) },
       INT32_C(           1),
      { EASYSIMD_FLOAT64_C(  -164.56), EASYSIMD_FLOAT64_C(  -635.52) } },
    { { EASYSIMD_FLOAT64_C(   516.16), EASYSIMD_FLOAT64_C(   674.11) },
      UINT8_C(205),
      { EASYSIMD_FLOAT64_C(  -471.29), EASYSIMD_FLOAT64_C(  -765.72) },
      { EASYSIMD_FLOAT64_C(  -833.83), EASYSIMD_FLOAT64_C(  -535.27) },
       INT32_C(          10),
      { EASYSIMD_FLOAT64_C(   471.29), EASYSIMD_FLOAT64_C(  -765.72) } },
    { { EASYSIMD_FLOAT64_C(  -636.12), EASYSIMD_FLOAT64_C(  -587.71) },
      UINT8_C(118),
      { EASYSIMD_FLOAT64_C(  -855.13), EASYSIMD_FLOAT64_C(  -574.73) },
      { EASYSIMD_FLOAT64_C(    38.07), EASYSIMD_FLOAT64_C(   159.73) },
       INT32_C(           0),
      { EASYSIMD_FLOAT64_C(  -636.12), EASYSIMD_FLOAT64_C(  -574.73) } },
  };

  easysimd__m128d src, a, b, r;

  src = easysimd_mm_loadu_pd(test_vec[0].src);
  a = easysimd_mm_loadu_pd(test_vec[0].a);
  b = easysimd_mm_loadu_pd(test_vec[0].b);
  r = easysimd_mm_mask_range_sd(src, test_vec[0].k, a, b, INT32_C(           4));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[0].r), 1);

  src = easysimd_mm_loadu_pd(test_vec[1].src);
  a = easysimd_mm_loadu_pd(test_vec[1].a);
  b = easysimd_mm_loadu_pd(test_vec[1].b);
  r = easysimd_mm_mask_range_sd(src, test_vec[1].k, a, b, INT32_C(           2));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[1].r), 1);

  src = easysimd_mm_loadu_pd(test_vec[2].src);
  a = easysimd_mm_loadu_pd(test_vec[2].a);
  b = easysimd_mm_loadu_pd(test_vec[2].b);
  r = easysimd_mm_mask_range_sd(src, test_vec[2].k, a, b, INT32_C(          14));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[2].r), 1);

  src = easysimd_mm_loadu_pd(test_vec[3].src);
  a = easysimd_mm_loadu_pd(test_vec[3].a);
  b = easysimd_mm_loadu_pd(test_vec[3].b);
  r = easysimd_mm_mask_range_sd(src, test_vec[3].k, a, b, INT32_C(          13));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[3].r), 1);

  src = easysimd_mm_loadu_pd(test_vec[4].src);
  a = easysimd_mm_loadu_pd(test_vec[4].a);
  b = easysimd_mm_loadu_pd(test_vec[4].b);
  r = easysimd_mm_mask_range_sd(src, test_vec[4].k, a, b, INT32_C(           5));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[4].r), 1);

  src = easysimd_mm_loadu_pd(test_vec[5].src);
  a = easysimd_mm_loadu_pd(test_vec[5].a);
  b = easysimd_mm_loadu_pd(test_vec[5].b);
  r = easysimd_mm_mask_range_sd(src, test_vec[5].k, a, b, INT32_C(           1));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[5].r), 1);

  src = easysimd_mm_loadu_pd(test_vec[6].src);
  a = easysimd_mm_loadu_pd(test_vec[6].a);
  b = easysimd_mm_loadu_pd(test_vec[6].b);
  r = easysimd_mm_mask_range_sd(src, test_vec[6].k, a, b, INT32_C(          10));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[6].r), 1);

  src = easysimd_mm_loadu_pd(test_vec[7].src);
  a = easysimd_mm_loadu_pd(test_vec[7].a);
  b = easysimd_mm_loadu_pd(test_vec[7].b);
  r = easysimd_mm_mask_range_sd(src, test_vec[7].k, a, b, INT32_C(           0));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d src = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    int imm8 = easysimd_test_codegen_rand() & 15;
    easysimd__m128d r;
    EASYSIMD_CONSTIFY_16_(easysimd_mm_mask_range_sd, r, easysimd_mm_setzero_pd(), imm8, src, k, a, b);

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
test_easysimd_mm_maskz_range_sd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const easysimd_float64 a[2];
    const easysimd_float64 b[2];
    const int imm8;
    const easysimd_float64 r[2];
  } test_vec[] = {
    { UINT8_C(210),
      { EASYSIMD_FLOAT64_C(   247.38), EASYSIMD_FLOAT64_C(    54.48) },
      { EASYSIMD_FLOAT64_C(  -758.53), EASYSIMD_FLOAT64_C(  -268.38) },
       INT32_C(           3),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    54.48) } },
    { UINT8_C(116),
      { EASYSIMD_FLOAT64_C(   567.05), EASYSIMD_FLOAT64_C(  -973.11) },
      { EASYSIMD_FLOAT64_C(   661.06), EASYSIMD_FLOAT64_C(  -144.91) },
       INT32_C(           9),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -973.11) } },
    { UINT8_C(137),
      { EASYSIMD_FLOAT64_C(  -653.59), EASYSIMD_FLOAT64_C(   364.64) },
      { EASYSIMD_FLOAT64_C(   883.80), EASYSIMD_FLOAT64_C(  -979.48) },
       INT32_C(           8),
      { EASYSIMD_FLOAT64_C(   653.59), EASYSIMD_FLOAT64_C(   364.64) } },
    { UINT8_C( 86),
      { EASYSIMD_FLOAT64_C(  -745.20), EASYSIMD_FLOAT64_C(   926.95) },
      { EASYSIMD_FLOAT64_C(  -122.76), EASYSIMD_FLOAT64_C(   852.52) },
       INT32_C(           7),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   926.95) } },
    { UINT8_C( 90),
      { EASYSIMD_FLOAT64_C(   195.09), EASYSIMD_FLOAT64_C(  -564.30) },
      { EASYSIMD_FLOAT64_C(   714.79), EASYSIMD_FLOAT64_C(  -766.84) },
       INT32_C(          11),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -564.30) } },
    { UINT8_C(171),
      { EASYSIMD_FLOAT64_C(   694.17), EASYSIMD_FLOAT64_C(  -157.20) },
      { EASYSIMD_FLOAT64_C(   748.14), EASYSIMD_FLOAT64_C(   935.63) },
       INT32_C(           4),
      { EASYSIMD_FLOAT64_C(   694.17), EASYSIMD_FLOAT64_C(  -157.20) } },
    { UINT8_C( 70),
      { EASYSIMD_FLOAT64_C(   527.38), EASYSIMD_FLOAT64_C(   141.48) },
      { EASYSIMD_FLOAT64_C(   742.18), EASYSIMD_FLOAT64_C(   188.44) },
       INT32_C(           3),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   141.48) } },
    { UINT8_C(166),
      { EASYSIMD_FLOAT64_C(   556.09), EASYSIMD_FLOAT64_C(  -657.03) },
      { EASYSIMD_FLOAT64_C(   498.20), EASYSIMD_FLOAT64_C(   439.89) },
       INT32_C(           1),
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -657.03) } },
  };

  easysimd__m128d  a, b, r;

  a = easysimd_mm_loadu_pd(test_vec[0].a);
  b = easysimd_mm_loadu_pd(test_vec[0].b);
  r = easysimd_mm_maskz_range_sd(test_vec[0].k, a, b, INT32_C(           3));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[0].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[1].a);
  b = easysimd_mm_loadu_pd(test_vec[1].b);
  r = easysimd_mm_maskz_range_sd(test_vec[1].k, a, b, INT32_C(           9));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[1].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[2].a);
  b = easysimd_mm_loadu_pd(test_vec[2].b);
  r = easysimd_mm_maskz_range_sd(test_vec[2].k, a, b, INT32_C(           8));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[2].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[3].a);
  b = easysimd_mm_loadu_pd(test_vec[3].b);
  r = easysimd_mm_maskz_range_sd(test_vec[3].k, a, b, INT32_C(           7));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[3].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[4].a);
  b = easysimd_mm_loadu_pd(test_vec[4].b);
  r = easysimd_mm_maskz_range_sd(test_vec[4].k, a, b, INT32_C(          11));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[4].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[5].a);
  b = easysimd_mm_loadu_pd(test_vec[5].b);
  r = easysimd_mm_maskz_range_sd(test_vec[5].k, a, b, INT32_C(           4));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[5].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[6].a);
  b = easysimd_mm_loadu_pd(test_vec[6].b);
  r = easysimd_mm_maskz_range_sd(test_vec[6].k, a, b, INT32_C(           3));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[6].r), 1);

  a = easysimd_mm_loadu_pd(test_vec[7].a);
  b = easysimd_mm_loadu_pd(test_vec[7].b);
  r = easysimd_mm_maskz_range_sd(test_vec[7].k, a, b, INT32_C(           1));
  easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[7].r), 1);

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m128d b = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    int imm8 = easysimd_test_codegen_rand() & 15;
    easysimd__m128d r;
    EASYSIMD_CONSTIFY_16_(easysimd_mm_maskz_range_sd, r, easysimd_mm_setzero_pd(), imm8, k, a, b);

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
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_range_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_range_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_range_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_range_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_range_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_range_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_range_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_range_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_range_ps)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_range_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_range_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_range_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_range_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_range_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_range_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_range_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_range_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_range_pd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(x_mm_range_ss)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_range_ss)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_range_ss)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(x_mm_range_sd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_range_sd)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_range_sd)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
