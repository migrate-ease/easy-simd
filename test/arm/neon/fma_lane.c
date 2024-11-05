#define EASYSIMD_TEST_ARM_NEON_INSN fma_lane

#include "test-neon.h"
#include "../../../easysimd/arm/neon/fma_lane.h"

static int
test_easysimd_vfmad_lane_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64_t a;
    easysimd_float64_t b;
    easysimd_float64_t v[1];
    easysimd_float64_t r;
  } test_vec[] = {
    { EASYSIMD_FLOAT64_C(  -551.91),
      EASYSIMD_FLOAT64_C(   -88.98),
      { EASYSIMD_FLOAT64_C(  -739.25) },
      EASYSIMD_FLOAT64_C( 65226.56) },
    { EASYSIMD_FLOAT64_C(   976.90),
      EASYSIMD_FLOAT64_C(  -627.36),
      { EASYSIMD_FLOAT64_C(   686.54) },
      EASYSIMD_FLOAT64_C(-429730.83) },
    { EASYSIMD_FLOAT64_C(   187.07),
      EASYSIMD_FLOAT64_C(   241.14),
      { EASYSIMD_FLOAT64_C(   939.17) },
      EASYSIMD_FLOAT64_C(226658.52) },
    { EASYSIMD_FLOAT64_C(   252.34),
      EASYSIMD_FLOAT64_C(   299.82),
      { EASYSIMD_FLOAT64_C(  -987.26) },
      EASYSIMD_FLOAT64_C(-295747.95) },
    { EASYSIMD_FLOAT64_C(   407.64),
      EASYSIMD_FLOAT64_C(   173.41),
      { EASYSIMD_FLOAT64_C(  -310.88) },
      EASYSIMD_FLOAT64_C(-53502.06) },
    { EASYSIMD_FLOAT64_C(  -130.57),
      EASYSIMD_FLOAT64_C(   -33.95),
      { EASYSIMD_FLOAT64_C(   448.04) },
      EASYSIMD_FLOAT64_C(-15341.53) },
    { EASYSIMD_FLOAT64_C(  -511.27),
      EASYSIMD_FLOAT64_C(  -230.11),
      { EASYSIMD_FLOAT64_C(  -658.57) },
      EASYSIMD_FLOAT64_C(151032.27) },
    { EASYSIMD_FLOAT64_C(   647.26),
      EASYSIMD_FLOAT64_C(  -290.71),
      { EASYSIMD_FLOAT64_C(   731.29) },
      EASYSIMD_FLOAT64_C(-211946.06) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64_t r = easysimd_vfmad_lane_f64(test_vec[i].a, test_vec[i].b, easysimd_vld1_f64(test_vec[i].v), 0);
    easysimd_assert_equal_f64(r, test_vec[i].r, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64_t a = easysimd_test_codegen_random_f64(-1000, 1000);
    easysimd_float64_t b = easysimd_test_codegen_random_f64(-1000, 1000);
    easysimd_float64x1_t v = easysimd_test_arm_neon_random_f64x1(-1000.0f, 1000.0f);
    easysimd_float64_t r = easysimd_vfmad_lane_f64(a, b, v, 0);

    easysimd_test_codegen_write_f64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f64(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x1(2, v, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_f64(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vfmad_laneq_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64_t a;
    easysimd_float64_t b;
    easysimd_float64_t v[2];
    easysimd_float64_t r0;
    easysimd_float64_t r1;
  } test_vec[] = {
    { EASYSIMD_FLOAT64_C(   604.01),
      EASYSIMD_FLOAT64_C(   734.02),
      { EASYSIMD_FLOAT64_C(  -923.33), EASYSIMD_FLOAT64_C(   297.94) },
      EASYSIMD_FLOAT64_C(-677138.68),
      EASYSIMD_FLOAT64_C(219297.93) },
    { EASYSIMD_FLOAT64_C(  -907.99),
      EASYSIMD_FLOAT64_C(   986.75),
      { EASYSIMD_FLOAT64_C(   501.64), EASYSIMD_FLOAT64_C(   955.09) },
      EASYSIMD_FLOAT64_C(494085.28),
      EASYSIMD_FLOAT64_C(941527.07) },
    { EASYSIMD_FLOAT64_C(    13.97),
      EASYSIMD_FLOAT64_C(  -914.34),
      { EASYSIMD_FLOAT64_C(  -964.88), EASYSIMD_FLOAT64_C(   825.04) },
      EASYSIMD_FLOAT64_C(882242.35),
      EASYSIMD_FLOAT64_C(-754353.10) },
    { EASYSIMD_FLOAT64_C(   459.76),
      EASYSIMD_FLOAT64_C(   412.19),
      { EASYSIMD_FLOAT64_C(  -359.87), EASYSIMD_FLOAT64_C(  -176.18) },
      EASYSIMD_FLOAT64_C(-147875.06),
      EASYSIMD_FLOAT64_C(-72159.87) },
    { EASYSIMD_FLOAT64_C(   695.71),
      EASYSIMD_FLOAT64_C(   444.70),
      { EASYSIMD_FLOAT64_C(   640.70), EASYSIMD_FLOAT64_C(   160.57) },
      EASYSIMD_FLOAT64_C(285615.00),
      EASYSIMD_FLOAT64_C( 72101.19) },
    { EASYSIMD_FLOAT64_C(   627.74),
      EASYSIMD_FLOAT64_C(  -709.72),
      { EASYSIMD_FLOAT64_C(   507.32), EASYSIMD_FLOAT64_C(  -251.68) },
      EASYSIMD_FLOAT64_C(-359427.41),
      EASYSIMD_FLOAT64_C(179250.07) },
    { EASYSIMD_FLOAT64_C(  -175.69),
      EASYSIMD_FLOAT64_C(  -701.50),
      { EASYSIMD_FLOAT64_C(   957.39), EASYSIMD_FLOAT64_C(   849.93) },
      EASYSIMD_FLOAT64_C(-671784.78),
      EASYSIMD_FLOAT64_C(-596401.58) },
    { EASYSIMD_FLOAT64_C(   354.47),
      EASYSIMD_FLOAT64_C(  -561.80),
      { EASYSIMD_FLOAT64_C(   783.03), EASYSIMD_FLOAT64_C(   -41.52) },
      EASYSIMD_FLOAT64_C(-439551.78),
      EASYSIMD_FLOAT64_C( 23680.41) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t v = easysimd_vld1q_f64(test_vec[i].v);
    easysimd_float64_t r0 = easysimd_vfmad_laneq_f64(test_vec[i].a, test_vec[i].b, v, 0);
    easysimd_float64_t r1 = easysimd_vfmad_laneq_f64(test_vec[i].a, test_vec[i].b, v, 1);
    easysimd_assert_equal_f64(r0, test_vec[i].r0, 1);
    easysimd_assert_equal_f64(r1, test_vec[i].r1, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64_t a = easysimd_test_codegen_random_f64(-1000, 1000);
    easysimd_float64_t b = easysimd_test_codegen_random_f64(-1000, 1000);
    easysimd_float64x2_t v = easysimd_test_arm_neon_random_f64x2(-1000.0, 1000.0);
    easysimd_float64_t r0 = easysimd_vfmad_laneq_f64(a, b, v, 0);
    easysimd_float64_t r1 = easysimd_vfmad_laneq_f64(a, b, v, 1);

    easysimd_test_codegen_write_f64(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f64(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, v, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_f64(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_f64(2, r1, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vfmas_lane_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32_t a;
    easysimd_float32_t b;
    easysimd_float32_t v[2];
    easysimd_float32_t r0;
    easysimd_float32_t r1;
  } test_vec[] = {
    { EASYSIMD_FLOAT32_C(  -827.78),
      EASYSIMD_FLOAT32_C(   859.69),
      { EASYSIMD_FLOAT32_C(  -743.58), EASYSIMD_FLOAT32_C(  -735.77) },
      EASYSIMD_FLOAT32_C(-640076.06),
      EASYSIMD_FLOAT32_C(-633361.94) },
    { EASYSIMD_FLOAT32_C(   846.44),
      EASYSIMD_FLOAT32_C(   758.06),
      { EASYSIMD_FLOAT32_C(  -780.68), EASYSIMD_FLOAT32_C(  -139.59) },
      EASYSIMD_FLOAT32_C(-590955.81),
      EASYSIMD_FLOAT32_C(-104971.15) },
    { EASYSIMD_FLOAT32_C(   843.73),
      EASYSIMD_FLOAT32_C(  -745.56),
      { EASYSIMD_FLOAT32_C(  -314.55), EASYSIMD_FLOAT32_C(   303.49) },
      EASYSIMD_FLOAT32_C(235359.62),
      EASYSIMD_FLOAT32_C(-225426.27) },
    { EASYSIMD_FLOAT32_C(   666.63),
      EASYSIMD_FLOAT32_C(   325.58),
      { EASYSIMD_FLOAT32_C(  -872.69), EASYSIMD_FLOAT32_C(   362.33) },
      EASYSIMD_FLOAT32_C(-283463.78),
      EASYSIMD_FLOAT32_C(118634.02) },
    { EASYSIMD_FLOAT32_C(  -229.72),
      EASYSIMD_FLOAT32_C(   768.01),
      { EASYSIMD_FLOAT32_C(  -477.09), EASYSIMD_FLOAT32_C(  -601.98) },
      EASYSIMD_FLOAT32_C(-366639.62),
      EASYSIMD_FLOAT32_C(-462556.38) },
    { EASYSIMD_FLOAT32_C(  -941.70),
      EASYSIMD_FLOAT32_C(  -969.77),
      { EASYSIMD_FLOAT32_C(   146.34), EASYSIMD_FLOAT32_C(  -117.39) },
      EASYSIMD_FLOAT32_C(-142857.84),
      EASYSIMD_FLOAT32_C(112899.60) },
    { EASYSIMD_FLOAT32_C(  -671.27),
      EASYSIMD_FLOAT32_C(   103.73),
      { EASYSIMD_FLOAT32_C(  -267.47), EASYSIMD_FLOAT32_C(   683.20) },
      EASYSIMD_FLOAT32_C(-28415.93),
      EASYSIMD_FLOAT32_C( 70197.07) },
    { EASYSIMD_FLOAT32_C(   541.93),
      EASYSIMD_FLOAT32_C(  -484.44),
      { EASYSIMD_FLOAT32_C(  -358.32), EASYSIMD_FLOAT32_C(   714.15) },
      EASYSIMD_FLOAT32_C(174126.47),
      EASYSIMD_FLOAT32_C(-345420.91) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t v = easysimd_vld1_f32(test_vec[i].v);
    easysimd_float32_t r0 = easysimd_vfmas_lane_f32(test_vec[i].a, test_vec[i].b, v, 0);
    easysimd_float32_t r1 = easysimd_vfmas_lane_f32(test_vec[i].a, test_vec[i].b, v, 1);
    easysimd_assert_equal_f32(r0, test_vec[i].r0, 1);
    easysimd_assert_equal_f32(r1, test_vec[i].r1, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32_t a = easysimd_test_codegen_random_f32(-1000, 1000);
    easysimd_float32_t b = easysimd_test_codegen_random_f32(-1000, 1000);
    easysimd_float32x2_t v = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32_t r0 = easysimd_vfmas_lane_f32(a, b, v, 0);
    easysimd_float32_t r1 = easysimd_vfmas_lane_f32(a, b, v, 1);

    easysimd_test_codegen_write_f32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, v, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_f32(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_f32(2, r1, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vfmas_laneq_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32_t a;
    easysimd_float32_t b;
    easysimd_float32_t v[4];
    easysimd_float32_t r0;
    easysimd_float32_t r1;
    easysimd_float32_t r2;
    easysimd_float32_t r3;
  } test_vec[] = {
    { EASYSIMD_FLOAT32_C(  -624.75),
      EASYSIMD_FLOAT32_C(  -101.90),
      { EASYSIMD_FLOAT32_C(   978.37), EASYSIMD_FLOAT32_C(  -778.30), EASYSIMD_FLOAT32_C(  -343.84), EASYSIMD_FLOAT32_C(  -802.30) },
      EASYSIMD_FLOAT32_C(-100320.66),
      EASYSIMD_FLOAT32_C( 78684.02),
      EASYSIMD_FLOAT32_C( 34412.55),
      EASYSIMD_FLOAT32_C( 81129.62) },
    { EASYSIMD_FLOAT32_C(    82.11),
      EASYSIMD_FLOAT32_C(  -500.11),
      { EASYSIMD_FLOAT32_C(  -547.87), EASYSIMD_FLOAT32_C(   767.56), EASYSIMD_FLOAT32_C(   803.38), EASYSIMD_FLOAT32_C(  -881.24) },
      EASYSIMD_FLOAT32_C(274077.38),
      EASYSIMD_FLOAT32_C(-383782.31),
      EASYSIMD_FLOAT32_C(-401696.25),
      EASYSIMD_FLOAT32_C(440799.03) },
    { EASYSIMD_FLOAT32_C(    93.14),
      EASYSIMD_FLOAT32_C(   930.69),
      { EASYSIMD_FLOAT32_C(   481.10), EASYSIMD_FLOAT32_C(   863.42), EASYSIMD_FLOAT32_C(   698.70), EASYSIMD_FLOAT32_C(  -996.00) },
      EASYSIMD_FLOAT32_C(447848.09),
      EASYSIMD_FLOAT32_C(803669.50),
      EASYSIMD_FLOAT32_C(650366.25),
      EASYSIMD_FLOAT32_C(-926874.12) },
    { EASYSIMD_FLOAT32_C(  -738.56),
      EASYSIMD_FLOAT32_C(   757.00),
      { EASYSIMD_FLOAT32_C(  -965.77), EASYSIMD_FLOAT32_C(   407.79), EASYSIMD_FLOAT32_C(  -360.40), EASYSIMD_FLOAT32_C(  -637.04) },
      EASYSIMD_FLOAT32_C(-731826.44),
      EASYSIMD_FLOAT32_C(307958.47),
      EASYSIMD_FLOAT32_C(-273561.34),
      EASYSIMD_FLOAT32_C(-482977.81) },
    { EASYSIMD_FLOAT32_C(  -488.48),
      EASYSIMD_FLOAT32_C(   372.13),
      { EASYSIMD_FLOAT32_C(  -953.84), EASYSIMD_FLOAT32_C(  -946.55), EASYSIMD_FLOAT32_C(   887.69), EASYSIMD_FLOAT32_C(  -312.16) },
      EASYSIMD_FLOAT32_C(-355440.97),
      EASYSIMD_FLOAT32_C(-352728.12),
      EASYSIMD_FLOAT32_C(329847.59),
      EASYSIMD_FLOAT32_C(-116652.59) },
    { EASYSIMD_FLOAT32_C(   767.60),
      EASYSIMD_FLOAT32_C(  -737.05),
      { EASYSIMD_FLOAT32_C(   585.94), EASYSIMD_FLOAT32_C(   745.97), EASYSIMD_FLOAT32_C(  -515.36), EASYSIMD_FLOAT32_C(  -757.90) },
      EASYSIMD_FLOAT32_C(-431099.47),
      EASYSIMD_FLOAT32_C(-549049.56),
      EASYSIMD_FLOAT32_C(380613.66),
      EASYSIMD_FLOAT32_C(559377.81) },
    { EASYSIMD_FLOAT32_C(   943.67),
      EASYSIMD_FLOAT32_C(   566.75),
      { EASYSIMD_FLOAT32_C(  -258.01), EASYSIMD_FLOAT32_C(  -604.20), EASYSIMD_FLOAT32_C(   334.31), EASYSIMD_FLOAT32_C(  -454.64) },
      EASYSIMD_FLOAT32_C(-145283.50),
      EASYSIMD_FLOAT32_C(-341486.69),
      EASYSIMD_FLOAT32_C(190413.86),
      EASYSIMD_FLOAT32_C(-256723.56) },
    { EASYSIMD_FLOAT32_C(  -485.44),
      EASYSIMD_FLOAT32_C(  -572.55),
      { EASYSIMD_FLOAT32_C(  -523.95), EASYSIMD_FLOAT32_C(   995.66), EASYSIMD_FLOAT32_C(  -709.13), EASYSIMD_FLOAT32_C(  -825.25) },
      EASYSIMD_FLOAT32_C(299502.12),
      EASYSIMD_FLOAT32_C(-570550.56),
      EASYSIMD_FLOAT32_C(405526.94),
      EASYSIMD_FLOAT32_C(472011.44) }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t v = easysimd_vld1q_f32(test_vec[i].v);
    easysimd_float32_t r0 = easysimd_vfmas_laneq_f32(test_vec[i].a, test_vec[i].b, v, 0);
    easysimd_float32_t r1 = easysimd_vfmas_laneq_f32(test_vec[i].a, test_vec[i].b, v, 1);
    easysimd_float32_t r2 = easysimd_vfmas_laneq_f32(test_vec[i].a, test_vec[i].b, v, 2);
    easysimd_float32_t r3 = easysimd_vfmas_laneq_f32(test_vec[i].a, test_vec[i].b, v, 3);

    easysimd_assert_equal_f32(r0, test_vec[i].r0, 1);
    easysimd_assert_equal_f32(r1, test_vec[i].r1, 1);
    easysimd_assert_equal_f32(r2, test_vec[i].r2, 1);
    easysimd_assert_equal_f32(r3, test_vec[i].r3, 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32_t a = easysimd_test_codegen_random_f32(-1000, 1000);
    easysimd_float32_t b = easysimd_test_codegen_random_f32(-1000, 1000);
    easysimd_float32x4_t v = easysimd_test_arm_neon_random_f32x4(-1000.0, 1000.0);
    easysimd_float32_t r0 = easysimd_vfmas_laneq_f32(a, b, v, 0);
    easysimd_float32_t r1 = easysimd_vfmas_laneq_f32(a, b, v, 1);
    easysimd_float32_t r2 = easysimd_vfmas_laneq_f32(a, b, v, 2);
    easysimd_float32_t r3 = easysimd_vfmas_laneq_f32(a, b, v, 3);

    easysimd_test_codegen_write_f32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_codegen_write_f32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, v, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_f32(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_f32(2, r1, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_f32(2, r2, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_f32(2, r3, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vfma_lane_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32_t a[2];
    easysimd_float32_t b[2];
    easysimd_float32_t v[2];
    easysimd_float32_t r0[2];
    easysimd_float32_t r1[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   999.66), EASYSIMD_FLOAT32_C(  -447.69) },
      { EASYSIMD_FLOAT32_C(   931.75), EASYSIMD_FLOAT32_C(  -966.11) },
      { EASYSIMD_FLOAT32_C(   960.10), EASYSIMD_FLOAT32_C(  -428.65) },
      { EASYSIMD_FLOAT32_C(895572.81), EASYSIMD_FLOAT32_C(-928009.88) },
      { EASYSIMD_FLOAT32_C(-398394.97), EASYSIMD_FLOAT32_C(413675.34) } },
    { { EASYSIMD_FLOAT32_C(  -603.14), EASYSIMD_FLOAT32_C(  -528.38) },
      { EASYSIMD_FLOAT32_C(   943.48), EASYSIMD_FLOAT32_C(  -556.98) },
      { EASYSIMD_FLOAT32_C(  -474.92), EASYSIMD_FLOAT32_C(   831.18) },
      { EASYSIMD_FLOAT32_C(-448680.66), EASYSIMD_FLOAT32_C(263992.56) },
      { EASYSIMD_FLOAT32_C(783598.56), EASYSIMD_FLOAT32_C(-463479.00) } },
    { { EASYSIMD_FLOAT32_C(   130.86), EASYSIMD_FLOAT32_C(  -707.32) },
      { EASYSIMD_FLOAT32_C(  -905.88), EASYSIMD_FLOAT32_C(  -283.20) },
      { EASYSIMD_FLOAT32_C(  -961.35), EASYSIMD_FLOAT32_C(  -421.24) },
      { EASYSIMD_FLOAT32_C(870998.56), EASYSIMD_FLOAT32_C(271547.00) },
      { EASYSIMD_FLOAT32_C(381723.75), EASYSIMD_FLOAT32_C(118587.85) } },
    { { EASYSIMD_FLOAT32_C(   -41.10), EASYSIMD_FLOAT32_C(   982.32) },
      { EASYSIMD_FLOAT32_C(  -854.49), EASYSIMD_FLOAT32_C(   700.89) },
      { EASYSIMD_FLOAT32_C(  -621.88), EASYSIMD_FLOAT32_C(   479.82) },
      { EASYSIMD_FLOAT32_C(531349.12), EASYSIMD_FLOAT32_C(-434887.16) },
      { EASYSIMD_FLOAT32_C(-410042.50), EASYSIMD_FLOAT32_C(337283.38) } },
    { { EASYSIMD_FLOAT32_C(  -753.75), EASYSIMD_FLOAT32_C(  -107.31) },
      { EASYSIMD_FLOAT32_C(   907.27), EASYSIMD_FLOAT32_C(  -277.70) },
      { EASYSIMD_FLOAT32_C(  -111.65), EASYSIMD_FLOAT32_C(  -801.86) },
      { EASYSIMD_FLOAT32_C(-102050.45), EASYSIMD_FLOAT32_C( 30897.90) },
      { EASYSIMD_FLOAT32_C(-728257.25), EASYSIMD_FLOAT32_C(222569.22) } },
    { { EASYSIMD_FLOAT32_C(  -102.95), EASYSIMD_FLOAT32_C(  -111.99) },
      { EASYSIMD_FLOAT32_C(  -249.55), EASYSIMD_FLOAT32_C(  -171.20) },
      { EASYSIMD_FLOAT32_C(   -78.10), EASYSIMD_FLOAT32_C(  -289.45) },
      { EASYSIMD_FLOAT32_C( 19386.90), EASYSIMD_FLOAT32_C( 13258.73) },
      { EASYSIMD_FLOAT32_C( 72129.30), EASYSIMD_FLOAT32_C( 49441.85) } },
    { { EASYSIMD_FLOAT32_C(   400.15), EASYSIMD_FLOAT32_C(   318.76) },
      { EASYSIMD_FLOAT32_C(   182.18), EASYSIMD_FLOAT32_C(   343.63) },
      { EASYSIMD_FLOAT32_C(   761.79), EASYSIMD_FLOAT32_C(   707.25) },
      { EASYSIMD_FLOAT32_C(139183.05), EASYSIMD_FLOAT32_C(262092.66) },
      { EASYSIMD_FLOAT32_C(129246.95), EASYSIMD_FLOAT32_C(243351.08) } },
    { { EASYSIMD_FLOAT32_C(   174.81), EASYSIMD_FLOAT32_C(  -107.35) },
      { EASYSIMD_FLOAT32_C(   999.93), EASYSIMD_FLOAT32_C(   268.93) },
      { EASYSIMD_FLOAT32_C(   609.45), EASYSIMD_FLOAT32_C(  -961.42) },
      { EASYSIMD_FLOAT32_C(609582.12), EASYSIMD_FLOAT32_C(163792.03) },
      { EASYSIMD_FLOAT32_C(-961177.88), EASYSIMD_FLOAT32_C(-258662.02) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t v = easysimd_vld1_f32(test_vec[i].v);
    easysimd_float32x2_t a = easysimd_vld1_f32(test_vec[i].a);
    easysimd_float32x2_t b = easysimd_vld1_f32(test_vec[i].b);
    easysimd_float32x2_t r0 = easysimd_vfma_lane_f32(a, b, v, 0);
    easysimd_float32x2_t r1 = easysimd_vfma_lane_f32(a, b, v, 1);
    easysimd_test_arm_neon_assert_equal_f32x2(r0, easysimd_vld1_f32(test_vec[i].r0), 1);
    easysimd_test_arm_neon_assert_equal_f32x2(r1, easysimd_vld1_f32(test_vec[i].r1), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x2_t a = easysimd_test_arm_neon_random_f32x2(-1000, 1000);
    easysimd_float32x2_t b = easysimd_test_arm_neon_random_f32x2(-1000, 1000);
    easysimd_float32x2_t v = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x2_t r0 = easysimd_vfma_lane_f32(a, b, v, 0);
    easysimd_float32x2_t r1 = easysimd_vfma_lane_f32(a, b, v, 1);

    easysimd_test_arm_neon_write_f32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, v, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, r1, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vfma_lane_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64_t a[1];
    easysimd_float64_t b[1];
    easysimd_float64_t v[1];
    easysimd_float64_t r[1];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -347.72) },
      { EASYSIMD_FLOAT64_C(   254.71) },
      { EASYSIMD_FLOAT64_C(  -412.60) },
      { EASYSIMD_FLOAT64_C(-105441.07) } },
    { { EASYSIMD_FLOAT64_C(  -140.90) },
      { EASYSIMD_FLOAT64_C(   744.16) },
      { EASYSIMD_FLOAT64_C(   647.70) },
      { EASYSIMD_FLOAT64_C(481851.53) } },
    { { EASYSIMD_FLOAT64_C(  -884.03) },
      { EASYSIMD_FLOAT64_C(   355.58) },
      { EASYSIMD_FLOAT64_C(   910.36) },
      { EASYSIMD_FLOAT64_C(322821.78) } },
    { { EASYSIMD_FLOAT64_C(   119.64) },
      { EASYSIMD_FLOAT64_C(    31.17) },
      { EASYSIMD_FLOAT64_C(   -56.43) },
      { EASYSIMD_FLOAT64_C( -1639.28) } },
    { { EASYSIMD_FLOAT64_C(   348.34) },
      { EASYSIMD_FLOAT64_C(   549.72) },
      { EASYSIMD_FLOAT64_C(  -199.38) },
      { EASYSIMD_FLOAT64_C(-109254.83) } },
    { { EASYSIMD_FLOAT64_C(   782.79) },
      { EASYSIMD_FLOAT64_C(  -213.64) },
      { EASYSIMD_FLOAT64_C(   787.12) },
      { EASYSIMD_FLOAT64_C(-167377.53) } },
    { { EASYSIMD_FLOAT64_C(   -92.53) },
      { EASYSIMD_FLOAT64_C(   -51.83) },
      { EASYSIMD_FLOAT64_C(   255.77) },
      { EASYSIMD_FLOAT64_C(-13349.09) } },
    { { EASYSIMD_FLOAT64_C(  -738.23) },
      { EASYSIMD_FLOAT64_C(   610.13) },
      { EASYSIMD_FLOAT64_C(   896.76) },
      { EASYSIMD_FLOAT64_C(546401.95) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x1_t v = easysimd_vld1_f64(test_vec[i].v);
    easysimd_float64x1_t r = easysimd_vfma_lane_f64(easysimd_vld1_f64(test_vec[i].a), easysimd_vld1_f64(test_vec[i].b), v, 0);
    easysimd_test_arm_neon_assert_equal_f64x1(r, easysimd_vld1_f64(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x1_t a = easysimd_test_arm_neon_random_f64x1(-1000, 1000);
    easysimd_float64x1_t b = easysimd_test_arm_neon_random_f64x1(-1000, 1000);
    easysimd_float64x1_t v = easysimd_test_arm_neon_random_f64x1(-1000.0, 1000.0);
    easysimd_float64x1_t r = easysimd_vfma_lane_f64(a, b, v, 0);

    easysimd_test_arm_neon_write_f64x1(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x1(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x1(2, v, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x1(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vfma_laneq_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32_t a[2];
    easysimd_float32_t b[2];
    easysimd_float32_t v[4];
    easysimd_float32_t r0[2];
    easysimd_float32_t r1[2];
    easysimd_float32_t r2[2];
    easysimd_float32_t r3[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   847.69), EASYSIMD_FLOAT32_C(  -431.65) },
      { EASYSIMD_FLOAT32_C(  -979.10), EASYSIMD_FLOAT32_C(   993.21) },
      { EASYSIMD_FLOAT32_C(  -730.77), EASYSIMD_FLOAT32_C(  -600.98), EASYSIMD_FLOAT32_C(   473.02), EASYSIMD_FLOAT32_C(  -484.51) },
      { EASYSIMD_FLOAT32_C(716344.62), EASYSIMD_FLOAT32_C(-726239.75) },
      { EASYSIMD_FLOAT32_C(589267.19), EASYSIMD_FLOAT32_C(-597331.00) },
      { EASYSIMD_FLOAT32_C(-462286.16), EASYSIMD_FLOAT32_C(469376.53) },
      { EASYSIMD_FLOAT32_C(475231.44), EASYSIMD_FLOAT32_C(-481651.84) } },
    { { EASYSIMD_FLOAT32_C(   291.70), EASYSIMD_FLOAT32_C(   380.29) },
      { EASYSIMD_FLOAT32_C(   237.79), EASYSIMD_FLOAT32_C(  -819.95) },
      { EASYSIMD_FLOAT32_C(   578.43), EASYSIMD_FLOAT32_C(  -865.16), EASYSIMD_FLOAT32_C(    68.06), EASYSIMD_FLOAT32_C(  -671.12) },
      { EASYSIMD_FLOAT32_C(137836.56), EASYSIMD_FLOAT32_C(-473903.38) },
      { EASYSIMD_FLOAT32_C(-205434.69), EASYSIMD_FLOAT32_C(709768.25) },
      { EASYSIMD_FLOAT32_C( 16475.69), EASYSIMD_FLOAT32_C(-55425.50) },
      { EASYSIMD_FLOAT32_C(-159293.92), EASYSIMD_FLOAT32_C(550665.12) } },
    { { EASYSIMD_FLOAT32_C(   -36.36), EASYSIMD_FLOAT32_C(   989.96) },
      { EASYSIMD_FLOAT32_C(    39.43), EASYSIMD_FLOAT32_C(  -636.21) },
      { EASYSIMD_FLOAT32_C(   308.73), EASYSIMD_FLOAT32_C(  -778.40), EASYSIMD_FLOAT32_C(   707.42), EASYSIMD_FLOAT32_C(    70.51) },
      { EASYSIMD_FLOAT32_C( 12136.86), EASYSIMD_FLOAT32_C(-195427.17) },
      { EASYSIMD_FLOAT32_C(-30728.67), EASYSIMD_FLOAT32_C(496215.84) },
      { EASYSIMD_FLOAT32_C( 27857.21), EASYSIMD_FLOAT32_C(-449077.72) },
      { EASYSIMD_FLOAT32_C(  2743.85), EASYSIMD_FLOAT32_C(-43869.21) } },
    { { EASYSIMD_FLOAT32_C(   928.86), EASYSIMD_FLOAT32_C(  -117.77) },
      { EASYSIMD_FLOAT32_C(   963.16), EASYSIMD_FLOAT32_C(   928.78) },
      { EASYSIMD_FLOAT32_C(  -848.84), EASYSIMD_FLOAT32_C(   572.61), EASYSIMD_FLOAT32_C(   967.36), EASYSIMD_FLOAT32_C(   998.85) },
      { EASYSIMD_FLOAT32_C(-816639.88), EASYSIMD_FLOAT32_C(-788503.44) },
      { EASYSIMD_FLOAT32_C(552443.88), EASYSIMD_FLOAT32_C(531710.94) },
      { EASYSIMD_FLOAT32_C(932651.25), EASYSIMD_FLOAT32_C(898346.88) },
      { EASYSIMD_FLOAT32_C(962981.19), EASYSIMD_FLOAT32_C(927594.12) } },
    { { EASYSIMD_FLOAT32_C(  -859.04), EASYSIMD_FLOAT32_C(   988.25) },
      { EASYSIMD_FLOAT32_C(   992.06), EASYSIMD_FLOAT32_C(  -589.81) },
      { EASYSIMD_FLOAT32_C(  -612.73), EASYSIMD_FLOAT32_C(   465.08), EASYSIMD_FLOAT32_C(   -74.32), EASYSIMD_FLOAT32_C(   678.97) },
      { EASYSIMD_FLOAT32_C(-608723.94), EASYSIMD_FLOAT32_C(362382.53) },
      { EASYSIMD_FLOAT32_C(460528.22), EASYSIMD_FLOAT32_C(-273320.56) },
      { EASYSIMD_FLOAT32_C(-74588.94), EASYSIMD_FLOAT32_C( 44822.93) },
      { EASYSIMD_FLOAT32_C(672719.94), EASYSIMD_FLOAT32_C(-399475.03) } },
    { { EASYSIMD_FLOAT32_C(  -154.63), EASYSIMD_FLOAT32_C(  -836.54) },
      { EASYSIMD_FLOAT32_C(   859.02), EASYSIMD_FLOAT32_C(  -576.20) },
      { EASYSIMD_FLOAT32_C(  -701.70), EASYSIMD_FLOAT32_C(   -72.92), EASYSIMD_FLOAT32_C(  -247.32), EASYSIMD_FLOAT32_C(   261.94) },
      { EASYSIMD_FLOAT32_C(-602929.00), EASYSIMD_FLOAT32_C(403483.00) },
      { EASYSIMD_FLOAT32_C(-62794.37), EASYSIMD_FLOAT32_C( 41179.96) },
      { EASYSIMD_FLOAT32_C(-212607.47), EASYSIMD_FLOAT32_C(141669.25) },
      { EASYSIMD_FLOAT32_C(224857.08), EASYSIMD_FLOAT32_C(-151766.38) } },
    { { EASYSIMD_FLOAT32_C(   -82.96), EASYSIMD_FLOAT32_C(   792.10) },
      { EASYSIMD_FLOAT32_C(   625.73), EASYSIMD_FLOAT32_C(  -774.23) },
      { EASYSIMD_FLOAT32_C(  -986.29), EASYSIMD_FLOAT32_C(   333.15), EASYSIMD_FLOAT32_C(   296.28), EASYSIMD_FLOAT32_C(   942.56) },
      { EASYSIMD_FLOAT32_C(-617234.19), EASYSIMD_FLOAT32_C(764407.38) },
      { EASYSIMD_FLOAT32_C(208378.98), EASYSIMD_FLOAT32_C(-257142.61) },
      { EASYSIMD_FLOAT32_C(185308.31), EASYSIMD_FLOAT32_C(-228596.75) },
      { EASYSIMD_FLOAT32_C(589705.06), EASYSIMD_FLOAT32_C(-728966.12) } },
    { { EASYSIMD_FLOAT32_C(  -784.62), EASYSIMD_FLOAT32_C(   259.44) },
      { EASYSIMD_FLOAT32_C(   871.35), EASYSIMD_FLOAT32_C(  -633.46) },
      { EASYSIMD_FLOAT32_C(  -167.95), EASYSIMD_FLOAT32_C(   838.70), EASYSIMD_FLOAT32_C(  -634.61), EASYSIMD_FLOAT32_C(   -26.99) },
      { EASYSIMD_FLOAT32_C(-147127.84), EASYSIMD_FLOAT32_C(106649.05) },
      { EASYSIMD_FLOAT32_C(730016.62), EASYSIMD_FLOAT32_C(-531023.50) },
      { EASYSIMD_FLOAT32_C(-553752.00), EASYSIMD_FLOAT32_C(402259.50) },
      { EASYSIMD_FLOAT32_C(-24302.36), EASYSIMD_FLOAT32_C( 17356.53) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t v = easysimd_vld1q_f32(test_vec[i].v);
    easysimd_float32x2_t a = easysimd_vld1_f32(test_vec[i].a);
    easysimd_float32x2_t b = easysimd_vld1_f32(test_vec[i].b);
    easysimd_float32x2_t r0 = easysimd_vfma_laneq_f32(a, b, v, 0);
    easysimd_float32x2_t r1 = easysimd_vfma_laneq_f32(a, b, v, 1);
    easysimd_float32x2_t r2 = easysimd_vfma_laneq_f32(a, b, v, 2);
    easysimd_float32x2_t r3 = easysimd_vfma_laneq_f32(a, b, v, 3);
    easysimd_test_arm_neon_assert_equal_f32x2(r0, easysimd_vld1_f32(test_vec[i].r0), 1);
    easysimd_test_arm_neon_assert_equal_f32x2(r1, easysimd_vld1_f32(test_vec[i].r1), 1);
    easysimd_test_arm_neon_assert_equal_f32x2(r2, easysimd_vld1_f32(test_vec[i].r2), 1);
    easysimd_test_arm_neon_assert_equal_f32x2(r3, easysimd_vld1_f32(test_vec[i].r3), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x2_t a = easysimd_test_arm_neon_random_f32x2(-1000, 1000);
    easysimd_float32x2_t b = easysimd_test_arm_neon_random_f32x2(-1000, 1000);
    easysimd_float32x4_t v = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x2_t r0 = easysimd_vfma_laneq_f32(a, b, v, 0);
    easysimd_float32x2_t r1 = easysimd_vfma_laneq_f32(a, b, v, 1);
    easysimd_float32x2_t r2 = easysimd_vfma_laneq_f32(a, b, v, 2);
    easysimd_float32x2_t r3 = easysimd_vfma_laneq_f32(a, b, v, 3);

    easysimd_test_arm_neon_write_f32x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, v, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, r1, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, r2, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, r3, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vfma_laneq_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64_t a[1];
    easysimd_float64_t b[1];
    easysimd_float64_t v[2];
    easysimd_float64_t r0[1];
    easysimd_float64_t r1[1];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   826.96) },
      { EASYSIMD_FLOAT64_C(  -642.55) },
      { EASYSIMD_FLOAT64_C(   383.20), EASYSIMD_FLOAT64_C(  -785.77) },
      { EASYSIMD_FLOAT64_C(-245398.20) },
      { EASYSIMD_FLOAT64_C(505723.47) } },
    { { EASYSIMD_FLOAT64_C(   822.53) },
      { EASYSIMD_FLOAT64_C(  -691.12) },
      { EASYSIMD_FLOAT64_C(   893.20), EASYSIMD_FLOAT64_C(  -332.10) },
      { EASYSIMD_FLOAT64_C(-616485.85) },
      { EASYSIMD_FLOAT64_C(230343.48) } },
    { { EASYSIMD_FLOAT64_C(  -527.66) },
      { EASYSIMD_FLOAT64_C(   752.23) },
      { EASYSIMD_FLOAT64_C(    91.70), EASYSIMD_FLOAT64_C(  -229.35) },
      { EASYSIMD_FLOAT64_C( 68451.83) },
      { EASYSIMD_FLOAT64_C(-173051.61) } },
    { { EASYSIMD_FLOAT64_C(  -320.69) },
      { EASYSIMD_FLOAT64_C(   844.37) },
      { EASYSIMD_FLOAT64_C(  -967.41), EASYSIMD_FLOAT64_C(   596.35) },
      { EASYSIMD_FLOAT64_C(-817172.67) },
      { EASYSIMD_FLOAT64_C(503219.36) } },
    { { EASYSIMD_FLOAT64_C(   636.48) },
      { EASYSIMD_FLOAT64_C(   658.33) },
      { EASYSIMD_FLOAT64_C(   822.12), EASYSIMD_FLOAT64_C(   650.19) },
      { EASYSIMD_FLOAT64_C(541862.74) },
      { EASYSIMD_FLOAT64_C(428676.06) } },
    { { EASYSIMD_FLOAT64_C(    -8.52) },
      { EASYSIMD_FLOAT64_C(   118.40) },
      { EASYSIMD_FLOAT64_C(   592.75), EASYSIMD_FLOAT64_C(   206.86) },
      { EASYSIMD_FLOAT64_C( 70173.08) },
      { EASYSIMD_FLOAT64_C( 24483.70) } },
    { { EASYSIMD_FLOAT64_C(  -622.15) },
      { EASYSIMD_FLOAT64_C(   464.09) },
      { EASYSIMD_FLOAT64_C(   573.40), EASYSIMD_FLOAT64_C(   209.90) },
      { EASYSIMD_FLOAT64_C(265487.06) },
      { EASYSIMD_FLOAT64_C( 96790.34) } },
    { { EASYSIMD_FLOAT64_C(   302.80) },
      { EASYSIMD_FLOAT64_C(   938.79) },
      { EASYSIMD_FLOAT64_C(  -817.08), EASYSIMD_FLOAT64_C(   129.76) },
      { EASYSIMD_FLOAT64_C(-766763.73) },
      { EASYSIMD_FLOAT64_C(122120.19) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t v = easysimd_vld1q_f64(test_vec[i].v);
    easysimd_float64x1_t a = easysimd_vld1_f64(test_vec[i].a);
    easysimd_float64x1_t b = easysimd_vld1_f64(test_vec[i].b);
    easysimd_float64x1_t r0 = easysimd_vfma_laneq_f64(a, b, v, 0);
    easysimd_float64x1_t r1 = easysimd_vfma_laneq_f64(a, b, v, 1);
    easysimd_test_arm_neon_assert_equal_f64x1(r0, easysimd_vld1_f64(test_vec[i].r0), 1);
    easysimd_test_arm_neon_assert_equal_f64x1(r1, easysimd_vld1_f64(test_vec[i].r1), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x1_t a = easysimd_test_arm_neon_random_f64x1(-1000, 1000);
    easysimd_float64x1_t b = easysimd_test_arm_neon_random_f64x1(-1000, 1000);
    easysimd_float64x2_t v = easysimd_test_arm_neon_random_f64x2(-1000.0, 1000.0);
    easysimd_float64x1_t r0 = easysimd_vfma_laneq_f64(a, b, v, 0);
    easysimd_float64x1_t r1 = easysimd_vfma_laneq_f64(a, b, v, 1);

    easysimd_test_arm_neon_write_f64x1(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x1(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, v, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x1(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x1(2, r1, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vfmaq_lane_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32_t a[4];
    easysimd_float32_t b[4];
    easysimd_float32_t v[2];
    easysimd_float32_t r0[4];
    easysimd_float32_t r1[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -283.03), EASYSIMD_FLOAT32_C(   426.03), EASYSIMD_FLOAT32_C(   208.07), EASYSIMD_FLOAT32_C(   486.07) },
      { EASYSIMD_FLOAT32_C(   560.22), EASYSIMD_FLOAT32_C(  -645.61), EASYSIMD_FLOAT32_C(    78.84), EASYSIMD_FLOAT32_C(   -91.74) },
      { EASYSIMD_FLOAT32_C(  -682.90), EASYSIMD_FLOAT32_C(   898.67) },
      { EASYSIMD_FLOAT32_C(-382857.25), EASYSIMD_FLOAT32_C(441313.09), EASYSIMD_FLOAT32_C(-53631.77), EASYSIMD_FLOAT32_C( 63135.32) },
      { EASYSIMD_FLOAT32_C(503169.84), EASYSIMD_FLOAT32_C(-579764.31), EASYSIMD_FLOAT32_C( 71059.21), EASYSIMD_FLOAT32_C(-81957.91) } },
    { { EASYSIMD_FLOAT32_C(  -935.47), EASYSIMD_FLOAT32_C(  -943.56), EASYSIMD_FLOAT32_C(  -781.95), EASYSIMD_FLOAT32_C(  -357.85) },
      { EASYSIMD_FLOAT32_C(   351.18), EASYSIMD_FLOAT32_C(  -601.74), EASYSIMD_FLOAT32_C(   751.32), EASYSIMD_FLOAT32_C(  -953.07) },
      { EASYSIMD_FLOAT32_C(   724.74), EASYSIMD_FLOAT32_C(   545.92) },
      { EASYSIMD_FLOAT32_C(253578.72), EASYSIMD_FLOAT32_C(-437048.59), EASYSIMD_FLOAT32_C(543729.69), EASYSIMD_FLOAT32_C(-691085.81) },
      { EASYSIMD_FLOAT32_C(190780.70), EASYSIMD_FLOAT32_C(-329445.44), EASYSIMD_FLOAT32_C(409378.66), EASYSIMD_FLOAT32_C(-520657.81) } },
    { { EASYSIMD_FLOAT32_C(   913.42), EASYSIMD_FLOAT32_C(  -415.66), EASYSIMD_FLOAT32_C(   559.47), EASYSIMD_FLOAT32_C(  -534.15) },
      { EASYSIMD_FLOAT32_C(  -714.48), EASYSIMD_FLOAT32_C(   470.66), EASYSIMD_FLOAT32_C(  -993.66), EASYSIMD_FLOAT32_C(   491.27) },
      { EASYSIMD_FLOAT32_C(  -899.78), EASYSIMD_FLOAT32_C(   945.28) },
      { EASYSIMD_FLOAT32_C(643788.25), EASYSIMD_FLOAT32_C(-423906.12), EASYSIMD_FLOAT32_C(894634.88), EASYSIMD_FLOAT32_C(-442569.06) },
      { EASYSIMD_FLOAT32_C(-674470.25), EASYSIMD_FLOAT32_C(444489.84), EASYSIMD_FLOAT32_C(-938727.44), EASYSIMD_FLOAT32_C(463853.56) } },
    { { EASYSIMD_FLOAT32_C(    62.23), EASYSIMD_FLOAT32_C(  -182.82), EASYSIMD_FLOAT32_C(   371.31), EASYSIMD_FLOAT32_C(  -729.70) },
      { EASYSIMD_FLOAT32_C(  -696.75), EASYSIMD_FLOAT32_C(   -68.47), EASYSIMD_FLOAT32_C(  -375.31), EASYSIMD_FLOAT32_C(   382.09) },
      { EASYSIMD_FLOAT32_C(   839.79), EASYSIMD_FLOAT32_C(   -58.21) },
      { EASYSIMD_FLOAT32_C(-585061.44), EASYSIMD_FLOAT32_C(-57683.24), EASYSIMD_FLOAT32_C(-314810.25), EASYSIMD_FLOAT32_C(320145.66) },
      { EASYSIMD_FLOAT32_C( 40620.05), EASYSIMD_FLOAT32_C(  3802.82), EASYSIMD_FLOAT32_C( 22218.11), EASYSIMD_FLOAT32_C(-22971.16) } },
    { { EASYSIMD_FLOAT32_C(   280.76), EASYSIMD_FLOAT32_C(   904.32), EASYSIMD_FLOAT32_C(    -1.77), EASYSIMD_FLOAT32_C(   498.81) },
      { EASYSIMD_FLOAT32_C(  -453.54), EASYSIMD_FLOAT32_C(  -650.59), EASYSIMD_FLOAT32_C(   897.07), EASYSIMD_FLOAT32_C(  -702.22) },
      { EASYSIMD_FLOAT32_C(  -603.66), EASYSIMD_FLOAT32_C(   621.81) },
      { EASYSIMD_FLOAT32_C(274064.72), EASYSIMD_FLOAT32_C(393639.47), EASYSIMD_FLOAT32_C(-541527.00), EASYSIMD_FLOAT32_C(424400.91) },
      { EASYSIMD_FLOAT32_C(-281734.94), EASYSIMD_FLOAT32_C(-403639.06), EASYSIMD_FLOAT32_C(557805.31), EASYSIMD_FLOAT32_C(-436148.59) } },
    { { EASYSIMD_FLOAT32_C(   843.70), EASYSIMD_FLOAT32_C(  -690.24), EASYSIMD_FLOAT32_C(  -793.85), EASYSIMD_FLOAT32_C(   403.17) },
      { EASYSIMD_FLOAT32_C(  -224.39), EASYSIMD_FLOAT32_C(  -508.33), EASYSIMD_FLOAT32_C(  -126.17), EASYSIMD_FLOAT32_C(  -218.05) },
      { EASYSIMD_FLOAT32_C(   982.94), EASYSIMD_FLOAT32_C(   -25.96) },
      { EASYSIMD_FLOAT32_C(-219718.20), EASYSIMD_FLOAT32_C(-500348.12), EASYSIMD_FLOAT32_C(-124811.39), EASYSIMD_FLOAT32_C(-213926.91) },
      { EASYSIMD_FLOAT32_C(  6668.86), EASYSIMD_FLOAT32_C( 12506.01), EASYSIMD_FLOAT32_C(  2481.52), EASYSIMD_FLOAT32_C(  6063.75) } },
    { { EASYSIMD_FLOAT32_C(  -272.77), EASYSIMD_FLOAT32_C(    45.17), EASYSIMD_FLOAT32_C(   791.23), EASYSIMD_FLOAT32_C(  -901.46) },
      { EASYSIMD_FLOAT32_C(   315.47), EASYSIMD_FLOAT32_C(  -905.52), EASYSIMD_FLOAT32_C(    30.07), EASYSIMD_FLOAT32_C(   940.16) },
      { EASYSIMD_FLOAT32_C(   476.57), EASYSIMD_FLOAT32_C(  -130.15) },
      { EASYSIMD_FLOAT32_C(150070.77), EASYSIMD_FLOAT32_C(-431498.50), EASYSIMD_FLOAT32_C( 15121.69), EASYSIMD_FLOAT32_C(447150.59) },
      { EASYSIMD_FLOAT32_C(-41331.19), EASYSIMD_FLOAT32_C(117898.59), EASYSIMD_FLOAT32_C( -3122.38), EASYSIMD_FLOAT32_C(-123263.27) } },
    { { EASYSIMD_FLOAT32_C(  -118.04), EASYSIMD_FLOAT32_C(  -242.67), EASYSIMD_FLOAT32_C(  -225.83), EASYSIMD_FLOAT32_C(   880.19) },
      { EASYSIMD_FLOAT32_C(  -743.86), EASYSIMD_FLOAT32_C(   320.63), EASYSIMD_FLOAT32_C(  -770.40), EASYSIMD_FLOAT32_C(  -846.78) },
      { EASYSIMD_FLOAT32_C(   618.41), EASYSIMD_FLOAT32_C(  -374.06) },
      { EASYSIMD_FLOAT32_C(-460128.47), EASYSIMD_FLOAT32_C(198038.12), EASYSIMD_FLOAT32_C(-476648.88), EASYSIMD_FLOAT32_C(-522777.03) },
      { EASYSIMD_FLOAT32_C(278130.22), EASYSIMD_FLOAT32_C(-120177.53), EASYSIMD_FLOAT32_C(287950.00), EASYSIMD_FLOAT32_C(317626.72) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x2_t v = easysimd_vld1_f32(test_vec[i].v);
    easysimd_float32x4_t a = easysimd_vld1q_f32(test_vec[i].a);
    easysimd_float32x4_t b = easysimd_vld1q_f32(test_vec[i].b);
    easysimd_float32x4_t r0 = easysimd_vfmaq_lane_f32(a, b, v, 0);
    easysimd_float32x4_t r1 = easysimd_vfmaq_lane_f32(a, b, v, 1);
    easysimd_test_arm_neon_assert_equal_f32x4(r0, easysimd_vld1q_f32(test_vec[i].r0), 1);
    easysimd_test_arm_neon_assert_equal_f32x4(r1, easysimd_vld1q_f32(test_vec[i].r1), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x4_t a = easysimd_test_arm_neon_random_f32x4(-1000, 1000);
    easysimd_float32x4_t b = easysimd_test_arm_neon_random_f32x4(-1000, 1000);
    easysimd_float32x2_t v = easysimd_test_arm_neon_random_f32x2(-1000.0f, 1000.0f);
    easysimd_float32x4_t r0 = easysimd_vfmaq_lane_f32(a, b, v, 0);
    easysimd_float32x4_t r1 = easysimd_vfmaq_lane_f32(a, b, v, 1);

    easysimd_test_arm_neon_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x2(2, v, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, r1, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vfmaq_lane_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64_t a[2];
    easysimd_float64_t b[2];
    easysimd_float64_t v[1];
    easysimd_float64_t r0[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   775.03), EASYSIMD_FLOAT64_C(   462.11) },
      { EASYSIMD_FLOAT64_C(   -64.30), EASYSIMD_FLOAT64_C(   981.19) },
      { EASYSIMD_FLOAT64_C(  -134.72) },
      { EASYSIMD_FLOAT64_C(  9437.53), EASYSIMD_FLOAT64_C(-131723.81) } },
    { { EASYSIMD_FLOAT64_C(   711.30), EASYSIMD_FLOAT64_C(  -527.14) },
      { EASYSIMD_FLOAT64_C(   739.11), EASYSIMD_FLOAT64_C(  -506.74) },
      { EASYSIMD_FLOAT64_C(  -544.20) },
      { EASYSIMD_FLOAT64_C(-401512.36), EASYSIMD_FLOAT64_C(275240.77) } },
    { { EASYSIMD_FLOAT64_C(  -286.84), EASYSIMD_FLOAT64_C(   220.49) },
      { EASYSIMD_FLOAT64_C(   500.97), EASYSIMD_FLOAT64_C(  -495.62) },
      { EASYSIMD_FLOAT64_C(   319.03) },
      { EASYSIMD_FLOAT64_C(159537.62), EASYSIMD_FLOAT64_C(-157897.16) } },
    { { EASYSIMD_FLOAT64_C(  -183.56), EASYSIMD_FLOAT64_C(  -401.14) },
      { EASYSIMD_FLOAT64_C(  -650.91), EASYSIMD_FLOAT64_C(  -243.39) },
      { EASYSIMD_FLOAT64_C(  -924.57) },
      { EASYSIMD_FLOAT64_C(601628.30), EASYSIMD_FLOAT64_C(224629.95) } },
    { { EASYSIMD_FLOAT64_C(   218.95), EASYSIMD_FLOAT64_C(   638.56) },
      { EASYSIMD_FLOAT64_C(  -167.24), EASYSIMD_FLOAT64_C(   993.12) },
      { EASYSIMD_FLOAT64_C(   518.75) },
      { EASYSIMD_FLOAT64_C(-86536.80), EASYSIMD_FLOAT64_C(515819.56) } },
    { { EASYSIMD_FLOAT64_C(    88.91), EASYSIMD_FLOAT64_C(   313.75) },
      { EASYSIMD_FLOAT64_C(   748.35), EASYSIMD_FLOAT64_C(   242.12) },
      { EASYSIMD_FLOAT64_C(   -67.83) },
      { EASYSIMD_FLOAT64_C(-50671.67), EASYSIMD_FLOAT64_C(-16109.25) } },
    { { EASYSIMD_FLOAT64_C(  -625.71), EASYSIMD_FLOAT64_C(    17.15) },
      { EASYSIMD_FLOAT64_C(  -605.72), EASYSIMD_FLOAT64_C(   309.99) },
      { EASYSIMD_FLOAT64_C(    -1.66) },
      { EASYSIMD_FLOAT64_C(   379.79), EASYSIMD_FLOAT64_C(  -497.43) } },
    { { EASYSIMD_FLOAT64_C(   259.56), EASYSIMD_FLOAT64_C(    21.29) },
      { EASYSIMD_FLOAT64_C(   471.19), EASYSIMD_FLOAT64_C(    -1.33) },
      { EASYSIMD_FLOAT64_C(   514.55) },
      { EASYSIMD_FLOAT64_C(242710.37), EASYSIMD_FLOAT64_C(  -663.06) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x1_t v = easysimd_vld1_f64(test_vec[i].v);
    easysimd_float64x2_t a = easysimd_vld1q_f64(test_vec[i].a);
    easysimd_float64x2_t b = easysimd_vld1q_f64(test_vec[i].b);
    easysimd_float64x2_t r0 = easysimd_vfmaq_lane_f64(a, b, v, 0);
    easysimd_test_arm_neon_assert_equal_f64x2(r0, easysimd_vld1q_f64(test_vec[i].r0), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x2_t a = easysimd_test_arm_neon_random_f64x2(-1000, 1000);
    easysimd_float64x2_t b = easysimd_test_arm_neon_random_f64x2(-1000, 1000);
    easysimd_float64x1_t v = easysimd_test_arm_neon_random_f64x1(-1000.0, 1000.0);
    easysimd_float64x2_t r0 = easysimd_vfmaq_lane_f64(a, b, v, 0);

    easysimd_test_arm_neon_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x1(2, v, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, r0, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vfmaq_laneq_f32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float32_t a[4];
    easysimd_float32_t b[4];
    easysimd_float32_t v[4];
    easysimd_float32_t r0[4];
    easysimd_float32_t r1[4];
    easysimd_float32_t r2[4];
    easysimd_float32_t r3[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -703.76), EASYSIMD_FLOAT32_C(   566.12), EASYSIMD_FLOAT32_C(   343.98), EASYSIMD_FLOAT32_C(  -881.23) },
      { EASYSIMD_FLOAT32_C(   875.00), EASYSIMD_FLOAT32_C(   237.19), EASYSIMD_FLOAT32_C(  -213.33), EASYSIMD_FLOAT32_C(  -652.65) },
      { EASYSIMD_FLOAT32_C(   -10.59), EASYSIMD_FLOAT32_C(   878.37), EASYSIMD_FLOAT32_C(   117.99), EASYSIMD_FLOAT32_C(   668.72) },
      { EASYSIMD_FLOAT32_C( -9970.01), EASYSIMD_FLOAT32_C( -1945.72), EASYSIMD_FLOAT32_C(  2603.14), EASYSIMD_FLOAT32_C(  6030.33) },
      { EASYSIMD_FLOAT32_C(767870.00), EASYSIMD_FLOAT32_C(208906.70), EASYSIMD_FLOAT32_C(-187038.69), EASYSIMD_FLOAT32_C(-574149.44) },
      { EASYSIMD_FLOAT32_C(102537.48), EASYSIMD_FLOAT32_C( 28552.17), EASYSIMD_FLOAT32_C(-24826.83), EASYSIMD_FLOAT32_C(-77887.41) },
      { EASYSIMD_FLOAT32_C(584426.19), EASYSIMD_FLOAT32_C(159179.81), EASYSIMD_FLOAT32_C(-142314.05), EASYSIMD_FLOAT32_C(-437321.34) } },
    { { EASYSIMD_FLOAT32_C(   722.74), EASYSIMD_FLOAT32_C(   150.59), EASYSIMD_FLOAT32_C(   265.08), EASYSIMD_FLOAT32_C(   359.22) },
      { EASYSIMD_FLOAT32_C(  -191.09), EASYSIMD_FLOAT32_C(    87.20), EASYSIMD_FLOAT32_C(     9.41), EASYSIMD_FLOAT32_C(   800.39) },
      { EASYSIMD_FLOAT32_C(  -794.40), EASYSIMD_FLOAT32_C(  -397.84), EASYSIMD_FLOAT32_C(     7.25), EASYSIMD_FLOAT32_C(  -416.55) },
      { EASYSIMD_FLOAT32_C(152524.64), EASYSIMD_FLOAT32_C(-69121.09), EASYSIMD_FLOAT32_C( -7210.22), EASYSIMD_FLOAT32_C(-635470.62) },
      { EASYSIMD_FLOAT32_C( 76745.98), EASYSIMD_FLOAT32_C(-34541.05), EASYSIMD_FLOAT32_C( -3478.59), EASYSIMD_FLOAT32_C(-318067.94) },
      { EASYSIMD_FLOAT32_C(  -662.66), EASYSIMD_FLOAT32_C(   782.79), EASYSIMD_FLOAT32_C(   333.30), EASYSIMD_FLOAT32_C(  6162.05) },
      { EASYSIMD_FLOAT32_C( 80321.27), EASYSIMD_FLOAT32_C(-36172.57), EASYSIMD_FLOAT32_C( -3654.66), EASYSIMD_FLOAT32_C(-333043.22) } },
    { { EASYSIMD_FLOAT32_C(  -933.75), EASYSIMD_FLOAT32_C(  -419.35), EASYSIMD_FLOAT32_C(   793.35), EASYSIMD_FLOAT32_C(   369.05) },
      { EASYSIMD_FLOAT32_C(  -480.56), EASYSIMD_FLOAT32_C(   976.27), EASYSIMD_FLOAT32_C(  -501.19), EASYSIMD_FLOAT32_C(  -184.32) },
      { EASYSIMD_FLOAT32_C(   542.39), EASYSIMD_FLOAT32_C(   842.79), EASYSIMD_FLOAT32_C(   -65.55), EASYSIMD_FLOAT32_C(   417.39) },
      { EASYSIMD_FLOAT32_C(-261584.69), EASYSIMD_FLOAT32_C(529099.75), EASYSIMD_FLOAT32_C(-271047.09), EASYSIMD_FLOAT32_C(-99604.28) },
      { EASYSIMD_FLOAT32_C(-405944.91), EASYSIMD_FLOAT32_C(822371.25), EASYSIMD_FLOAT32_C(-421604.56), EASYSIMD_FLOAT32_C(-154974.00) },
      { EASYSIMD_FLOAT32_C( 30566.96), EASYSIMD_FLOAT32_C(-64413.85), EASYSIMD_FLOAT32_C( 33646.36), EASYSIMD_FLOAT32_C( 12451.23) },
      { EASYSIMD_FLOAT32_C(-201514.69), EASYSIMD_FLOAT32_C(407066.00), EASYSIMD_FLOAT32_C(-208398.36), EASYSIMD_FLOAT32_C(-76564.28) } },
    { { EASYSIMD_FLOAT32_C(    79.98), EASYSIMD_FLOAT32_C(   721.12), EASYSIMD_FLOAT32_C(   764.73), EASYSIMD_FLOAT32_C(  -930.61) },
      { EASYSIMD_FLOAT32_C(   599.49), EASYSIMD_FLOAT32_C(  -117.27), EASYSIMD_FLOAT32_C(   738.11), EASYSIMD_FLOAT32_C(   322.23) },
      { EASYSIMD_FLOAT32_C(  -966.68), EASYSIMD_FLOAT32_C(     3.19), EASYSIMD_FLOAT32_C(  -318.54), EASYSIMD_FLOAT32_C(  -157.77) },
      { EASYSIMD_FLOAT32_C(-579435.00), EASYSIMD_FLOAT32_C(114083.68), EASYSIMD_FLOAT32_C(-712751.44), EASYSIMD_FLOAT32_C(-312423.91) },
      { EASYSIMD_FLOAT32_C(  1992.35), EASYSIMD_FLOAT32_C(   347.03), EASYSIMD_FLOAT32_C(  3119.30), EASYSIMD_FLOAT32_C(    97.30) },
      { EASYSIMD_FLOAT32_C(-190881.56), EASYSIMD_FLOAT32_C( 38076.30), EASYSIMD_FLOAT32_C(-234352.83), EASYSIMD_FLOAT32_C(-103573.76) },
      { EASYSIMD_FLOAT32_C(-94501.55), EASYSIMD_FLOAT32_C( 19222.81), EASYSIMD_FLOAT32_C(-115686.88), EASYSIMD_FLOAT32_C(-51768.84) } },
    { { EASYSIMD_FLOAT32_C(  -909.61), EASYSIMD_FLOAT32_C(   690.86), EASYSIMD_FLOAT32_C(  -357.38), EASYSIMD_FLOAT32_C(  -704.01) },
      { EASYSIMD_FLOAT32_C(  -706.98), EASYSIMD_FLOAT32_C(   649.88), EASYSIMD_FLOAT32_C(  -120.56), EASYSIMD_FLOAT32_C(  -640.73) },
      { EASYSIMD_FLOAT32_C(  -769.47), EASYSIMD_FLOAT32_C(  -327.20), EASYSIMD_FLOAT32_C(   728.32), EASYSIMD_FLOAT32_C(  -250.02) },
      { EASYSIMD_FLOAT32_C(543090.25), EASYSIMD_FLOAT32_C(-499372.28), EASYSIMD_FLOAT32_C( 92409.91), EASYSIMD_FLOAT32_C(492318.47) },
      { EASYSIMD_FLOAT32_C(230414.25), EASYSIMD_FLOAT32_C(-211949.89), EASYSIMD_FLOAT32_C( 39089.85), EASYSIMD_FLOAT32_C(208942.84) },
      { EASYSIMD_FLOAT32_C(-515817.28), EASYSIMD_FLOAT32_C(474011.47), EASYSIMD_FLOAT32_C(-88163.64), EASYSIMD_FLOAT32_C(-467360.47) },
      { EASYSIMD_FLOAT32_C(175849.53), EASYSIMD_FLOAT32_C(-161792.14), EASYSIMD_FLOAT32_C( 29785.03), EASYSIMD_FLOAT32_C(159491.30) } },
    { { EASYSIMD_FLOAT32_C(  -350.93), EASYSIMD_FLOAT32_C(  -772.87), EASYSIMD_FLOAT32_C(   565.66), EASYSIMD_FLOAT32_C(  -808.54) },
      { EASYSIMD_FLOAT32_C(  -930.08), EASYSIMD_FLOAT32_C(  -499.89), EASYSIMD_FLOAT32_C(   608.85), EASYSIMD_FLOAT32_C(   149.90) },
      { EASYSIMD_FLOAT32_C(  -778.77), EASYSIMD_FLOAT32_C(   373.58), EASYSIMD_FLOAT32_C(   219.29), EASYSIMD_FLOAT32_C(   820.72) },
      { EASYSIMD_FLOAT32_C(723967.50), EASYSIMD_FLOAT32_C(388526.50), EASYSIMD_FLOAT32_C(-473588.44), EASYSIMD_FLOAT32_C(-117546.16) },
      { EASYSIMD_FLOAT32_C(-347810.22), EASYSIMD_FLOAT32_C(-187521.78), EASYSIMD_FLOAT32_C(228019.83), EASYSIMD_FLOAT32_C( 55191.10) },
      { EASYSIMD_FLOAT32_C(-204308.17), EASYSIMD_FLOAT32_C(-110393.75), EASYSIMD_FLOAT32_C(134080.36), EASYSIMD_FLOAT32_C( 32063.03) },
      { EASYSIMD_FLOAT32_C(-763686.19), EASYSIMD_FLOAT32_C(-411042.59), EASYSIMD_FLOAT32_C(500261.00), EASYSIMD_FLOAT32_C(122217.38) } },
    { { EASYSIMD_FLOAT32_C(  -743.69), EASYSIMD_FLOAT32_C(   -42.60), EASYSIMD_FLOAT32_C(   142.96), EASYSIMD_FLOAT32_C(  -710.38) },
      { EASYSIMD_FLOAT32_C(   960.59), EASYSIMD_FLOAT32_C(   824.41), EASYSIMD_FLOAT32_C(   131.85), EASYSIMD_FLOAT32_C(  -949.02) },
      { EASYSIMD_FLOAT32_C(   515.28), EASYSIMD_FLOAT32_C(   774.48), EASYSIMD_FLOAT32_C(  -653.03), EASYSIMD_FLOAT32_C(   808.30) },
      { EASYSIMD_FLOAT32_C(494229.16), EASYSIMD_FLOAT32_C(424759.41), EASYSIMD_FLOAT32_C( 68082.63), EASYSIMD_FLOAT32_C(-489721.44) },
      { EASYSIMD_FLOAT32_C(743214.06), EASYSIMD_FLOAT32_C(638446.44), EASYSIMD_FLOAT32_C(102258.15), EASYSIMD_FLOAT32_C(-735707.38) },
      { EASYSIMD_FLOAT32_C(-628037.81), EASYSIMD_FLOAT32_C(-538407.06), EASYSIMD_FLOAT32_C(-85959.05), EASYSIMD_FLOAT32_C(619028.19) },
      { EASYSIMD_FLOAT32_C(775701.19), EASYSIMD_FLOAT32_C(666328.00), EASYSIMD_FLOAT32_C(106717.32), EASYSIMD_FLOAT32_C(-767803.25) } },
    { { EASYSIMD_FLOAT32_C(   424.36), EASYSIMD_FLOAT32_C(   226.41), EASYSIMD_FLOAT32_C(  -832.43), EASYSIMD_FLOAT32_C(   654.89) },
      { EASYSIMD_FLOAT32_C(   899.21), EASYSIMD_FLOAT32_C(   895.89), EASYSIMD_FLOAT32_C(  -595.14), EASYSIMD_FLOAT32_C(  -451.72) },
      { EASYSIMD_FLOAT32_C(  -876.98), EASYSIMD_FLOAT32_C(   970.52), EASYSIMD_FLOAT32_C(  -260.27), EASYSIMD_FLOAT32_C(  -807.06) },
      { EASYSIMD_FLOAT32_C(-788164.81), EASYSIMD_FLOAT32_C(-785451.19), EASYSIMD_FLOAT32_C(521093.44), EASYSIMD_FLOAT32_C(396804.28) },
      { EASYSIMD_FLOAT32_C(873125.69), EASYSIMD_FLOAT32_C(869705.62), EASYSIMD_FLOAT32_C(-578427.75), EASYSIMD_FLOAT32_C(-437748.41) },
      { EASYSIMD_FLOAT32_C(-233613.02), EASYSIMD_FLOAT32_C(-232946.88), EASYSIMD_FLOAT32_C(154064.66), EASYSIMD_FLOAT32_C(118224.05) },
      { EASYSIMD_FLOAT32_C(-725292.06), EASYSIMD_FLOAT32_C(-722810.56), EASYSIMD_FLOAT32_C(479481.28), EASYSIMD_FLOAT32_C(365220.03) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float32x4_t v = easysimd_vld1q_f32(test_vec[i].v);
    easysimd_float32x4_t a = easysimd_vld1q_f32(test_vec[i].a);
    easysimd_float32x4_t b = easysimd_vld1q_f32(test_vec[i].b);
    easysimd_float32x4_t r0 = easysimd_vfmaq_laneq_f32(a, b, v, 0);
    easysimd_float32x4_t r1 = easysimd_vfmaq_laneq_f32(a, b, v, 1);
    easysimd_float32x4_t r2 = easysimd_vfmaq_laneq_f32(a, b, v, 2);
    easysimd_float32x4_t r3 = easysimd_vfmaq_laneq_f32(a, b, v, 3);
    easysimd_test_arm_neon_assert_equal_f32x4(r0, easysimd_vld1q_f32(test_vec[i].r0), 1);
    easysimd_test_arm_neon_assert_equal_f32x4(r1, easysimd_vld1q_f32(test_vec[i].r1), 1);
    easysimd_test_arm_neon_assert_equal_f32x4(r2, easysimd_vld1q_f32(test_vec[i].r2), 1);
    easysimd_test_arm_neon_assert_equal_f32x4(r3, easysimd_vld1q_f32(test_vec[i].r3), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float32x4_t a = easysimd_test_arm_neon_random_f32x4(-1000, 1000);
    easysimd_float32x4_t b = easysimd_test_arm_neon_random_f32x4(-1000, 1000);
    easysimd_float32x4_t v = easysimd_test_arm_neon_random_f32x4(-1000.0f, 1000.0f);
    easysimd_float32x4_t r0 = easysimd_vfmaq_laneq_f32(a, b, v, 0);
    easysimd_float32x4_t r1 = easysimd_vfmaq_laneq_f32(a, b, v, 1);
    easysimd_float32x4_t r2 = easysimd_vfmaq_laneq_f32(a, b, v, 2);
    easysimd_float32x4_t r3 = easysimd_vfmaq_laneq_f32(a, b, v, 3);

    easysimd_test_arm_neon_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, v, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, r1, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, r2, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f32x4(2, r3, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_vfmaq_laneq_f64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    easysimd_float64_t a[2];
    easysimd_float64_t b[2];
    easysimd_float64_t v[2];
    easysimd_float64_t r0[2];
    easysimd_float64_t r1[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -529.37), EASYSIMD_FLOAT64_C(  -651.42) },
      { EASYSIMD_FLOAT64_C(   342.84), EASYSIMD_FLOAT64_C(  -308.13) },
      { EASYSIMD_FLOAT64_C(   722.16), EASYSIMD_FLOAT64_C(  -437.87) },
      { EASYSIMD_FLOAT64_C(247055.96), EASYSIMD_FLOAT64_C(-223170.58) },
      { EASYSIMD_FLOAT64_C(-150648.72), EASYSIMD_FLOAT64_C(134269.46) } },
    { { EASYSIMD_FLOAT64_C(  -487.41), EASYSIMD_FLOAT64_C(   978.47) },
      { EASYSIMD_FLOAT64_C(   519.53), EASYSIMD_FLOAT64_C(   655.55) },
      { EASYSIMD_FLOAT64_C(  -731.91), EASYSIMD_FLOAT64_C(   480.12) },
      { EASYSIMD_FLOAT64_C(-380736.61), EASYSIMD_FLOAT64_C(-478825.13) },
      { EASYSIMD_FLOAT64_C(248949.33), EASYSIMD_FLOAT64_C(315721.14) } },
    { { EASYSIMD_FLOAT64_C(   479.96), EASYSIMD_FLOAT64_C(   399.95) },
      { EASYSIMD_FLOAT64_C(   531.09), EASYSIMD_FLOAT64_C(    -4.76) },
      { EASYSIMD_FLOAT64_C(   174.42), EASYSIMD_FLOAT64_C(   878.06) },
      { EASYSIMD_FLOAT64_C( 93112.68), EASYSIMD_FLOAT64_C(  -430.29) },
      { EASYSIMD_FLOAT64_C(466808.85), EASYSIMD_FLOAT64_C( -3779.62) } },
    { { EASYSIMD_FLOAT64_C(  -196.47), EASYSIMD_FLOAT64_C(  -401.22) },
      { EASYSIMD_FLOAT64_C(   104.48), EASYSIMD_FLOAT64_C(   -28.90) },
      { EASYSIMD_FLOAT64_C(  -746.33), EASYSIMD_FLOAT64_C(     3.69) },
      { EASYSIMD_FLOAT64_C(-78173.03), EASYSIMD_FLOAT64_C( 21167.72) },
      { EASYSIMD_FLOAT64_C(   189.06), EASYSIMD_FLOAT64_C(  -507.86) } },
    { { EASYSIMD_FLOAT64_C(  -133.00), EASYSIMD_FLOAT64_C(  -341.47) },
      { EASYSIMD_FLOAT64_C(   551.96), EASYSIMD_FLOAT64_C(    -9.98) },
      { EASYSIMD_FLOAT64_C(  -370.95), EASYSIMD_FLOAT64_C(  -708.30) },
      { EASYSIMD_FLOAT64_C(-204882.56), EASYSIMD_FLOAT64_C(  3360.61) },
      { EASYSIMD_FLOAT64_C(-391086.27), EASYSIMD_FLOAT64_C(  6727.36) } },
    { { EASYSIMD_FLOAT64_C(   182.97), EASYSIMD_FLOAT64_C(    99.68) },
      { EASYSIMD_FLOAT64_C(  -359.72), EASYSIMD_FLOAT64_C(  -474.19) },
      { EASYSIMD_FLOAT64_C(   791.55), EASYSIMD_FLOAT64_C(  -637.56) },
      { EASYSIMD_FLOAT64_C(-284553.40), EASYSIMD_FLOAT64_C(-375245.41) },
      { EASYSIMD_FLOAT64_C(229526.05), EASYSIMD_FLOAT64_C(302424.26) } },
    { { EASYSIMD_FLOAT64_C(    87.93), EASYSIMD_FLOAT64_C(  -695.86) },
      { EASYSIMD_FLOAT64_C(  -659.10), EASYSIMD_FLOAT64_C(  -392.54) },
      { EASYSIMD_FLOAT64_C(   959.69), EASYSIMD_FLOAT64_C(  -391.00) },
      { EASYSIMD_FLOAT64_C(-632443.75), EASYSIMD_FLOAT64_C(-377412.57) },
      { EASYSIMD_FLOAT64_C(257796.03), EASYSIMD_FLOAT64_C(152787.28) } },
    { { EASYSIMD_FLOAT64_C(  -912.43), EASYSIMD_FLOAT64_C(   439.65) },
      { EASYSIMD_FLOAT64_C(  -991.06), EASYSIMD_FLOAT64_C(   618.67) },
      { EASYSIMD_FLOAT64_C(  -565.12), EASYSIMD_FLOAT64_C(   183.37) },
      { EASYSIMD_FLOAT64_C(559155.40), EASYSIMD_FLOAT64_C(-349183.14) },
      { EASYSIMD_FLOAT64_C(-182643.10), EASYSIMD_FLOAT64_C(113885.17) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd_float64x2_t v = easysimd_vld1q_f64(test_vec[i].v);
    easysimd_float64x2_t a = easysimd_vld1q_f64(test_vec[i].a);
    easysimd_float64x2_t b = easysimd_vld1q_f64(test_vec[i].b);
    easysimd_float64x2_t r0 = easysimd_vfmaq_laneq_f64(a, b, v, 0);
    easysimd_float64x2_t r1 = easysimd_vfmaq_laneq_f64(a, b, v, 1);
    easysimd_test_arm_neon_assert_equal_f64x2(r0, easysimd_vld1q_f64(test_vec[i].r0), 1);
    easysimd_test_arm_neon_assert_equal_f64x2(r1, easysimd_vld1q_f64(test_vec[i].r1), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd_float64x2_t a = easysimd_test_arm_neon_random_f64x2(-1000, 1000);
    easysimd_float64x2_t b = easysimd_test_arm_neon_random_f64x2(-1000, 1000);
    easysimd_float64x2_t v = easysimd_test_arm_neon_random_f64x2(-1000.0, 1000.0);
    easysimd_float64x2_t r0 = easysimd_vfmaq_laneq_f64(a, b, v, 0);
    easysimd_float64x2_t r1 = easysimd_vfmaq_laneq_f64(a, b, v, 1);

    easysimd_test_arm_neon_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_arm_neon_write_f64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, v, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, r0, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_arm_neon_write_f64x2(2, r1, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(vfmad_lane_f64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vfmad_laneq_f64)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vfmas_lane_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vfmas_laneq_f32)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vfma_lane_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vfma_lane_f64)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vfma_laneq_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vfma_laneq_f64)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vfmaq_lane_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vfmaq_lane_f64)

EASYSIMD_TEST_FUNC_LIST_ENTRY(vfmaq_laneq_f32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(vfmaq_laneq_f64)
EASYSIMD_TEST_FUNC_LIST_END

#include "test-neon-footer.h"
