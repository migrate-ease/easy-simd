#define EASYSIMD_TEST_ARM_NEON_INSN cmla_rot180

#include "test-neon.h"
#include "../../../easysimd/arm/neon/cmla_rot180.h"

static int
test_easysimd_vcmla_rot180_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[4];
    easysimd_float32 b[4];
    easysimd_float32 r[4];
    easysimd_float32 r_[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   452.95), EASYSIMD_FLOAT32_C(   195.89) },
      { EASYSIMD_FLOAT32_C(   444.91), EASYSIMD_FLOAT32_C(    53.65) },
      { EASYSIMD_FLOAT32_C(  -340.35), EASYSIMD_FLOAT32_C(  -889.91) },
      { EASYSIMD_FLOAT32_C(-201862.34), EASYSIMD_FLOAT32_C(-25190.68) } },
    { { EASYSIMD_FLOAT32_C(   306.65), EASYSIMD_FLOAT32_C(  -419.57) },
      { EASYSIMD_FLOAT32_C(  -861.69), EASYSIMD_FLOAT32_C(   727.62) },
      { EASYSIMD_FLOAT32_C(  -997.28), EASYSIMD_FLOAT32_C(   144.31) },
      { EASYSIMD_FLOAT32_C(263239.97), EASYSIMD_FLOAT32_C(-222980.36) } },
    { { EASYSIMD_FLOAT32_C(   196.29), EASYSIMD_FLOAT32_C(   798.44) },
      { EASYSIMD_FLOAT32_C(  -778.96), EASYSIMD_FLOAT32_C(   915.31) },
      { EASYSIMD_FLOAT32_C(   222.88), EASYSIMD_FLOAT32_C(   691.69) },
      { EASYSIMD_FLOAT32_C(153124.94), EASYSIMD_FLOAT32_C(-178974.50) } },
    { { EASYSIMD_FLOAT32_C(   -68.29), EASYSIMD_FLOAT32_C(   627.45) },
      { EASYSIMD_FLOAT32_C(  -373.33), EASYSIMD_FLOAT32_C(   724.33) },
      { EASYSIMD_FLOAT32_C(   133.16), EASYSIMD_FLOAT32_C(   234.76) },
      { EASYSIMD_FLOAT32_C(-25361.54), EASYSIMD_FLOAT32_C( 49699.26) } },
    { { EASYSIMD_FLOAT32_C(   964.74), EASYSIMD_FLOAT32_C(   624.02) },
      { EASYSIMD_FLOAT32_C(  -835.66), EASYSIMD_FLOAT32_C(   908.14) },
      { EASYSIMD_FLOAT32_C(   913.15), EASYSIMD_FLOAT32_C(   697.78) },
      { EASYSIMD_FLOAT32_C(807107.75), EASYSIMD_FLOAT32_C(-875421.19) } },
    { { EASYSIMD_FLOAT32_C(   162.44), EASYSIMD_FLOAT32_C(   366.10) },
      { EASYSIMD_FLOAT32_C(  -106.33), EASYSIMD_FLOAT32_C(  -392.64) },
      { EASYSIMD_FLOAT32_C(  -580.25), EASYSIMD_FLOAT32_C(   553.32) },
      { EASYSIMD_FLOAT32_C( 16692.00), EASYSIMD_FLOAT32_C( 64333.77) } },
    { { EASYSIMD_FLOAT32_C(  -282.55), EASYSIMD_FLOAT32_C(   726.40) },
      { EASYSIMD_FLOAT32_C(  -866.24), EASYSIMD_FLOAT32_C(  -144.24) },
      { EASYSIMD_FLOAT32_C(   454.02), EASYSIMD_FLOAT32_C(  -863.52) },
      { EASYSIMD_FLOAT32_C(-244302.08), EASYSIMD_FLOAT32_C(-41618.53) } },
    { { EASYSIMD_FLOAT32_C(  -999.93), EASYSIMD_FLOAT32_C(  -349.69) },
      { EASYSIMD_FLOAT32_C(   934.92), EASYSIMD_FLOAT32_C(  -778.90) },
      { EASYSIMD_FLOAT32_C(  -434.38), EASYSIMD_FLOAT32_C(   157.80) },
      { EASYSIMD_FLOAT32_C(934420.12), EASYSIMD_FLOAT32_C(-778687.69) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t a = easysimd_vld1_f32(test_vec[i].a);
    easysimd_float32x2_t b = easysimd_vld1_f32(test_vec[i].b);
    easysimd_float32x2_t r = easysimd_vld1_f32(test_vec[i].r);
    easysimd_float32x2_t r_ = easysimd_vcmla_rot180_f32(r, a, b);

    easysimd_test_arm_neon_assert_equal_f32x2(r_, easysimd_vld1_f32(test_vec[i].r_), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x2_t a = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t b = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t r = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t r_ = easysimd_vcmla_rot180_f32(r, a, b);

    easysimd_test_arm_neon_write_f32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, r_, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vcmlaq_rot180_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32 a[4];
    easysimd_float32 b[4];
    easysimd_float32 r[4];
    easysimd_float32 r_[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   581.15), EASYSIMD_FLOAT32_C(  -956.71), EASYSIMD_FLOAT32_C(   565.79), EASYSIMD_FLOAT32_C(  -665.78) },
      { EASYSIMD_FLOAT32_C(   634.03), EASYSIMD_FLOAT32_C(  -372.68), EASYSIMD_FLOAT32_C(  -843.04), EASYSIMD_FLOAT32_C(  -573.31) },
      { EASYSIMD_FLOAT32_C(   743.16), EASYSIMD_FLOAT32_C(   262.51), EASYSIMD_FLOAT32_C(   315.71), EASYSIMD_FLOAT32_C(   643.97) },
      { EASYSIMD_FLOAT32_C(-367723.41), EASYSIMD_FLOAT32_C(216845.50), EASYSIMD_FLOAT32_C(477299.28), EASYSIMD_FLOAT32_C(325017.03) } },
    { { EASYSIMD_FLOAT32_C(   -14.23), EASYSIMD_FLOAT32_C(   146.05), EASYSIMD_FLOAT32_C(  -713.83), EASYSIMD_FLOAT32_C(   -20.06) },
      { EASYSIMD_FLOAT32_C(   104.47), EASYSIMD_FLOAT32_C(   216.28), EASYSIMD_FLOAT32_C(  -177.08), EASYSIMD_FLOAT32_C(    12.18) },
      { EASYSIMD_FLOAT32_C(  -643.47), EASYSIMD_FLOAT32_C(   708.88), EASYSIMD_FLOAT32_C(   686.41), EASYSIMD_FLOAT32_C(   709.88) },
      { EASYSIMD_FLOAT32_C(   843.14), EASYSIMD_FLOAT32_C(  3786.54), EASYSIMD_FLOAT32_C(-125718.61), EASYSIMD_FLOAT32_C(  9404.33) } },
    { { EASYSIMD_FLOAT32_C(  -814.88), EASYSIMD_FLOAT32_C(  -711.88), EASYSIMD_FLOAT32_C(  -748.81), EASYSIMD_FLOAT32_C(   858.53) },
      { EASYSIMD_FLOAT32_C(   737.87), EASYSIMD_FLOAT32_C(  -589.28), EASYSIMD_FLOAT32_C(  -895.36), EASYSIMD_FLOAT32_C(   319.02) },
      { EASYSIMD_FLOAT32_C(  -546.00), EASYSIMD_FLOAT32_C(   670.42), EASYSIMD_FLOAT32_C(   653.24), EASYSIMD_FLOAT32_C(  -911.97) },
      { EASYSIMD_FLOAT32_C(600729.50), EASYSIMD_FLOAT32_C(-479522.09), EASYSIMD_FLOAT32_C(-669801.25), EASYSIMD_FLOAT32_C(237973.39) } },
    { { EASYSIMD_FLOAT32_C(  -702.25), EASYSIMD_FLOAT32_C(   810.19), EASYSIMD_FLOAT32_C(  -485.28), EASYSIMD_FLOAT32_C(  -959.09) },
      { EASYSIMD_FLOAT32_C(    72.70), EASYSIMD_FLOAT32_C(   830.43), EASYSIMD_FLOAT32_C(   684.88), EASYSIMD_FLOAT32_C(  -941.53) },
      { EASYSIMD_FLOAT32_C(   -23.52), EASYSIMD_FLOAT32_C(   971.04), EASYSIMD_FLOAT32_C(    38.42), EASYSIMD_FLOAT32_C(  -919.05) },
      { EASYSIMD_FLOAT32_C( 51030.05), EASYSIMD_FLOAT32_C(584140.50), EASYSIMD_FLOAT32_C(332397.00), EASYSIMD_FLOAT32_C(-457824.75) } },
    { { EASYSIMD_FLOAT32_C(   187.33), EASYSIMD_FLOAT32_C(   861.34), EASYSIMD_FLOAT32_C(    93.13), EASYSIMD_FLOAT32_C(   543.85) },
      { EASYSIMD_FLOAT32_C(   570.21), EASYSIMD_FLOAT32_C(  -220.46), EASYSIMD_FLOAT32_C(   253.73), EASYSIMD_FLOAT32_C(   755.33) },
      { EASYSIMD_FLOAT32_C(    67.66), EASYSIMD_FLOAT32_C(   504.92), EASYSIMD_FLOAT32_C(   613.86), EASYSIMD_FLOAT32_C(  -194.47) },
      { EASYSIMD_FLOAT32_C(-106749.78), EASYSIMD_FLOAT32_C( 41803.70), EASYSIMD_FLOAT32_C(-23016.01), EASYSIMD_FLOAT32_C(-70538.35) } },
    { { EASYSIMD_FLOAT32_C(   915.64), EASYSIMD_FLOAT32_C(   718.50), EASYSIMD_FLOAT32_C(  -875.45), EASYSIMD_FLOAT32_C(  -630.36) },
      { EASYSIMD_FLOAT32_C(   388.92), EASYSIMD_FLOAT32_C(   777.79), EASYSIMD_FLOAT32_C(  -542.33), EASYSIMD_FLOAT32_C(   686.66) },
      { EASYSIMD_FLOAT32_C(   587.98), EASYSIMD_FLOAT32_C(   -27.61), EASYSIMD_FLOAT32_C(   727.57), EASYSIMD_FLOAT32_C(  -339.32) },
      { EASYSIMD_FLOAT32_C(-355522.75), EASYSIMD_FLOAT32_C(-712203.25), EASYSIMD_FLOAT32_C(-474055.25), EASYSIMD_FLOAT32_C(600797.19) } },
    { { EASYSIMD_FLOAT32_C(  -197.18), EASYSIMD_FLOAT32_C(   412.45), EASYSIMD_FLOAT32_C(  -280.84), EASYSIMD_FLOAT32_C(   779.30) },
      { EASYSIMD_FLOAT32_C(   383.50), EASYSIMD_FLOAT32_C(   757.57), EASYSIMD_FLOAT32_C(   860.25), EASYSIMD_FLOAT32_C(  -429.18) },
      { EASYSIMD_FLOAT32_C(   618.91), EASYSIMD_FLOAT32_C(   -46.62), EASYSIMD_FLOAT32_C(  -885.33), EASYSIMD_FLOAT32_C(   189.12) },
      { EASYSIMD_FLOAT32_C( 76237.44), EASYSIMD_FLOAT32_C(149331.03), EASYSIMD_FLOAT32_C(240707.28), EASYSIMD_FLOAT32_C(-120341.79) } },
    { { EASYSIMD_FLOAT32_C(   732.92), EASYSIMD_FLOAT32_C(   368.40), EASYSIMD_FLOAT32_C(   -55.55), EASYSIMD_FLOAT32_C(  -199.42) },
      { EASYSIMD_FLOAT32_C(  -126.68), EASYSIMD_FLOAT32_C(  -441.69), EASYSIMD_FLOAT32_C(   606.11), EASYSIMD_FLOAT32_C(  -211.04) },
      { EASYSIMD_FLOAT32_C(  -723.20), EASYSIMD_FLOAT32_C(   730.66), EASYSIMD_FLOAT32_C(   158.60), EASYSIMD_FLOAT32_C(   665.72) },
      { EASYSIMD_FLOAT32_C( 92123.10), EASYSIMD_FLOAT32_C(324454.09), EASYSIMD_FLOAT32_C( 33828.01), EASYSIMD_FLOAT32_C(-11057.55) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t a = easysimd_vld1q_f32(test_vec[i].a);
    easysimd_float32x4_t b = easysimd_vld1q_f32(test_vec[i].b);
    easysimd_float32x4_t r = easysimd_vld1q_f32(test_vec[i].r);
    easysimd_float32x4_t r_ = easysimd_vcmlaq_rot180_f32(r, a, b);
    easysimd_test_arm_neon_assert_equal_f32x4(r_, easysimd_vld1q_f32(test_vec[i].r_), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x4_t a = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t b = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t r = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t r_ = easysimd_vcmlaq_rot180_f32(r, a, b);

    easysimd_test_arm_neon_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, r_, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vcmlaq_rot180_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64 a[2];
    easysimd_float64 b[2];
    easysimd_float64 r[2];
    easysimd_float64 r_[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   262.77), EASYSIMD_FLOAT64_C(  -594.87) },
      { EASYSIMD_FLOAT64_C(  -382.77), EASYSIMD_FLOAT64_C(   813.86) },
      { EASYSIMD_FLOAT64_C(   154.31), EASYSIMD_FLOAT64_C(   621.90) },
      { EASYSIMD_FLOAT64_C(100734.78), EASYSIMD_FLOAT64_C(-213236.09) } },
    { { EASYSIMD_FLOAT64_C(   972.42), EASYSIMD_FLOAT64_C(  -414.89) },
      { EASYSIMD_FLOAT64_C(   356.78), EASYSIMD_FLOAT64_C(   498.91) },
      { EASYSIMD_FLOAT64_C(   960.93), EASYSIMD_FLOAT64_C(  -942.16) },
      { EASYSIMD_FLOAT64_C(-345979.08), EASYSIMD_FLOAT64_C(-486092.22) } },
    { { EASYSIMD_FLOAT64_C(   920.42), EASYSIMD_FLOAT64_C(  -123.39) },
      { EASYSIMD_FLOAT64_C(  -106.64), EASYSIMD_FLOAT64_C(   660.02) },
      { EASYSIMD_FLOAT64_C(   429.81), EASYSIMD_FLOAT64_C(   808.37) },
      { EASYSIMD_FLOAT64_C( 98583.40), EASYSIMD_FLOAT64_C(-606687.24) } },
    { { EASYSIMD_FLOAT64_C(   -29.61), EASYSIMD_FLOAT64_C(    98.57) },
      { EASYSIMD_FLOAT64_C(   828.43), EASYSIMD_FLOAT64_C(  -489.72) },
      { EASYSIMD_FLOAT64_C(   575.48), EASYSIMD_FLOAT64_C(   187.12) },
      { EASYSIMD_FLOAT64_C( 25105.29), EASYSIMD_FLOAT64_C(-14313.49) } },
    { { EASYSIMD_FLOAT64_C(  -389.68), EASYSIMD_FLOAT64_C(  -827.75) },
      { EASYSIMD_FLOAT64_C(    15.22), EASYSIMD_FLOAT64_C(   408.31) },
      { EASYSIMD_FLOAT64_C(   -24.59), EASYSIMD_FLOAT64_C(   768.89) },
      { EASYSIMD_FLOAT64_C(  5906.34), EASYSIMD_FLOAT64_C(159879.13) } },
    { { EASYSIMD_FLOAT64_C(  -685.93), EASYSIMD_FLOAT64_C(  -761.82) },
      { EASYSIMD_FLOAT64_C(  -825.99), EASYSIMD_FLOAT64_C(   -68.70) },
      { EASYSIMD_FLOAT64_C(  -947.96), EASYSIMD_FLOAT64_C(   328.32) },
      { EASYSIMD_FLOAT64_C(-567519.28), EASYSIMD_FLOAT64_C(-46795.07) } },
    { { EASYSIMD_FLOAT64_C(  -446.80), EASYSIMD_FLOAT64_C(  -975.54) },
      { EASYSIMD_FLOAT64_C(   913.43), EASYSIMD_FLOAT64_C(   909.98) },
      { EASYSIMD_FLOAT64_C(   523.37), EASYSIMD_FLOAT64_C(   874.37) },
      { EASYSIMD_FLOAT64_C(408643.89), EASYSIMD_FLOAT64_C(407453.43) } },
    { { EASYSIMD_FLOAT64_C(   967.82), EASYSIMD_FLOAT64_C(   443.78) },
      { EASYSIMD_FLOAT64_C(  -249.02), EASYSIMD_FLOAT64_C(  -138.82) },
      { EASYSIMD_FLOAT64_C(   103.81), EASYSIMD_FLOAT64_C(  -819.21) },
      { EASYSIMD_FLOAT64_C(241110.35), EASYSIMD_FLOAT64_C(133533.56) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t a = easysimd_vld1q_f64(test_vec[i].a);
    easysimd_float64x2_t b = easysimd_vld1q_f64(test_vec[i].b);
    easysimd_float64x2_t r = easysimd_vld1q_f64(test_vec[i].r);
    easysimd_float64x2_t r_ = easysimd_vcmlaq_rot180_f64(r, a, b);
    easysimd_test_arm_neon_assert_equal_f64x2(r_, easysimd_vld1q_f64(test_vec[i].r_), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x2_t a = easysimd_test_arm_neon_random_f64x2(-1000.0f, 1000.0f);
    easysimd_float64x2_t b = easysimd_test_arm_neon_random_f64x2(-1000.0f, 1000.0f);
    easysimd_float64x2_t r = easysimd_test_arm_neon_random_f64x2(-1000.0f, 1000.0f);
    easysimd_float64x2_t r_ = easysimd_vcmlaq_rot180_f64(r, a, b);

    easysimd_test_arm_neon_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, r_, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcmla_rot180_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcmlaq_rot180_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vcmlaq_rot180_f64)
EASYSIMD_TEST_FUNC_LIST_END

#include "test-neon-footer.h"
