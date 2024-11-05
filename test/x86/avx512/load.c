#define EASYSIMD_TEST_X86_AVX512_INSN load

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/load.h>

static int
test_easysimd_mm_mask_load_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[4];
    const uint8_t k;
    EASYSIMD_ALIGN_LIKE_64(easysimd__m128i) const int32_t a[4];
    const int32_t r[4];
  } test_vec[] = {
    { {  INT32_C(  1538202211), -INT32_C(  1195941574), -INT32_C(  1301749465),  INT32_C(   802173233) },
      UINT8_C( 23),
      { -INT32_C(   687222344),  INT32_C(  1571978287), -INT32_C(   587356156),  INT32_C(  1933524313) },
      { -INT32_C(   687222344),  INT32_C(  1571978287), -INT32_C(   587356156),  INT32_C(   802173233) } },
    { {  INT32_C(  1303223020),  INT32_C(   863266129), -INT32_C(    10213426), -INT32_C(  1357409289) },
      UINT8_C(100),
      { -INT32_C(  1651276256), -INT32_C(   526258120),  INT32_C(   725188077),  INT32_C(  1444392380) },
      {  INT32_C(  1303223020),  INT32_C(   863266129),  INT32_C(   725188077), -INT32_C(  1357409289) } },
    { { -INT32_C(  1062705830),  INT32_C(     9362393), -INT32_C(   755462849), -INT32_C(   986273884) },
      UINT8_C( 46),
      { -INT32_C(  1184472375), -INT32_C(  2136520957),  INT32_C(   758960768),  INT32_C(  1317573353) },
      { -INT32_C(  1062705830), -INT32_C(  2136520957),  INT32_C(   758960768),  INT32_C(  1317573353) } },
    { {  INT32_C(   354961466),  INT32_C(  1666459862), -INT32_C(   938990048),  INT32_C(   636931420) },
      UINT8_C( 47),
      { -INT32_C(  1556947108),  INT32_C(  1478800006), -INT32_C(  2143202834),  INT32_C(   565940441) },
      { -INT32_C(  1556947108),  INT32_C(  1478800006), -INT32_C(  2143202834),  INT32_C(   565940441) } },
    { { -INT32_C(   520630088),  INT32_C(  1258380069),  INT32_C(   816367971),  INT32_C(   459263423) },
      UINT8_C(172),
      {  INT32_C(  1127399313),  INT32_C(   875662051),  INT32_C(  1544466892),  INT32_C(  1024733036) },
      { -INT32_C(   520630088),  INT32_C(  1258380069),  INT32_C(  1544466892),  INT32_C(  1024733036) } },
    { { -INT32_C(  2107444185), -INT32_C(  1092243979),  INT32_C(   595400021),  INT32_C(    97491316) },
      UINT8_C( 88),
      { -INT32_C(  1942271998),  INT32_C(   710438777), -INT32_C(  1382566787), -INT32_C(  1110125368) },
      { -INT32_C(  2107444185), -INT32_C(  1092243979),  INT32_C(   595400021), -INT32_C(  1110125368) } },
    { { -INT32_C(   474851786),  INT32_C(  1345941819), -INT32_C(  2017174290),  INT32_C(   769640747) },
      UINT8_C( 17),
      { -INT32_C(  1970619878), -INT32_C(   955796462),  INT32_C(   529511499), -INT32_C(   564835192) },
      { -INT32_C(  1970619878),  INT32_C(  1345941819), -INT32_C(  2017174290),  INT32_C(   769640747) } },
    { {  INT32_C(  1880701439), -INT32_C(   832673422), -INT32_C(   151394771), -INT32_C(   536401979) },
      UINT8_C(225),
      {  INT32_C(  1190357649),  INT32_C(   647084657), -INT32_C(  1766936246),  INT32_C(  1066765574) },
      {  INT32_C(  1190357649), -INT32_C(   832673422), -INT32_C(   151394771), -INT32_C(   536401979) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_load_epi32(src, k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_load_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_mask_load_epi32(src, k, &a);

    easysimd_test_x86_write_i32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_load_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    EASYSIMD_ALIGN_LIKE_64(easysimd__m128i) const int32_t a[4];
    const int32_t r[4];
  } test_vec[] = {
    { UINT8_C(216),
      { -INT32_C(    69467487),  INT32_C(   699794915),  INT32_C(   314546872), -INT32_C(   826054755) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   826054755) } },
    { UINT8_C(129),
      { -INT32_C(   193286182),  INT32_C(  1140263748), -INT32_C(  2120891202),  INT32_C(   371354900) },
      { -INT32_C(   193286182),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 73),
      { -INT32_C(   749799139), -INT32_C(   496171485), -INT32_C(   599193064), -INT32_C(  2001286439) },
      { -INT32_C(   749799139),  INT32_C(           0),  INT32_C(           0), -INT32_C(  2001286439) } },
    { UINT8_C( 69),
      { -INT32_C(  1571763029), -INT32_C(  1519972592), -INT32_C(  1307411568), -INT32_C(  1949344623) },
      { -INT32_C(  1571763029),  INT32_C(           0), -INT32_C(  1307411568),  INT32_C(           0) } },
    { UINT8_C(170),
      {  INT32_C(   263302818), -INT32_C(   652621424),  INT32_C(  1554248357),  INT32_C(  1191700603) },
      {  INT32_C(           0), -INT32_C(   652621424),  INT32_C(           0),  INT32_C(  1191700603) } },
    { UINT8_C( 57),
      {  INT32_C(   273176489),  INT32_C(   260823292),  INT32_C(  1500192138),  INT32_C(  1459295656) },
      {  INT32_C(   273176489),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1459295656) } },
    { UINT8_C(198),
      {  INT32_C(   596698634),  INT32_C(  1645622719), -INT32_C(  1739877999),  INT32_C(   809665752) },
      {  INT32_C(           0),  INT32_C(  1645622719), -INT32_C(  1739877999),  INT32_C(           0) } },
    { UINT8_C(204),
      { -INT32_C(   559666094), -INT32_C(  1493488069), -INT32_C(  2101566585), -INT32_C(   544439559) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(  2101566585), -INT32_C(   544439559) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_load_epi32(k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_load_epi32");
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_maskz_load_epi32(k, &a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_load_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[2];
    const uint8_t k;
    EASYSIMD_ALIGN_LIKE_64(easysimd__m128i) const int64_t a[2];
    const int64_t r[2];
  } test_vec[] = {
    { {  INT64_C( 5349572000139511206), -INT64_C( 3638363984571190151) },
      UINT8_C( 89),
      {  INT64_C( 6159219652884632436), -INT64_C( 6137390733459714145) },
      {  INT64_C( 6159219652884632436), -INT64_C( 3638363984571190151) } },
    { {  INT64_C( 6072356570464707405), -INT64_C( 2616207214388386456) },
      UINT8_C(112),
      { -INT64_C( 7922840531748255108), -INT64_C( 1175753909898626618) },
      {  INT64_C( 6072356570464707405), -INT64_C( 2616207214388386456) } },
    { { -INT64_C( 6435251615513478160),  INT64_C( 3873631937630046452) },
      UINT8_C(241),
      { -INT64_C( 1727966432761437483), -INT64_C( 8065415777757141186) },
      { -INT64_C( 1727966432761437483),  INT64_C( 3873631937630046452) } },
    { {  INT64_C( 1390195063062092530),  INT64_C( 3664806794889108227) },
      UINT8_C(165),
      { -INT64_C( 2407670467502247599),  INT64_C( 2451511567036167984) },
      { -INT64_C( 2407670467502247599),  INT64_C( 3664806794889108227) } },
    { {  INT64_C( 4814137611816623317), -INT64_C( 1678205938888164851) },
      UINT8_C( 66),
      { -INT64_C( 8253523795538001911), -INT64_C( 1593298395444442998) },
      {  INT64_C( 4814137611816623317), -INT64_C( 1593298395444442998) } },
    { {  INT64_C( 4175631251280015268), -INT64_C( 4574609699255614864) },
      UINT8_C(248),
      {  INT64_C(  170394043519272252), -INT64_C(   13940367928824654) },
      {  INT64_C( 4175631251280015268), -INT64_C( 4574609699255614864) } },
    { { -INT64_C( 6319474254127647991),  INT64_C( 5492456474726600072) },
      UINT8_C( 37),
      { -INT64_C( 1723011152274776312), -INT64_C( 4488620917446393806) },
      { -INT64_C( 1723011152274776312),  INT64_C( 5492456474726600072) } },
    { {  INT64_C( 7706801206506983861),  INT64_C( 8553398295445779822) },
      UINT8_C(229),
      {  INT64_C( 3542163510008076883),  INT64_C( 1976517882275681410) },
      {  INT64_C( 3542163510008076883),  INT64_C( 8553398295445779822) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_mm_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_load_epi64(src, k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_load_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i64x2();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_mask_load_epi64(src, k, &a);

    easysimd_test_x86_write_i64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_load_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    EASYSIMD_ALIGN_LIKE_64(easysimd__m128i) const int64_t a[2];
    const int64_t r[2];
  } test_vec[] = {
    { UINT8_C( 18),
      {  INT64_C( 5428764998174809775), -INT64_C( 8428878817219629198) },
      {  INT64_C(                   0), -INT64_C( 8428878817219629198) } },
    { UINT8_C( 63),
      {  INT64_C( 2991221855264557028), -INT64_C(  526148912647734410) },
      {  INT64_C( 2991221855264557028), -INT64_C(  526148912647734410) } },
    { UINT8_C(  6),
      {  INT64_C( 1512455505348458871), -INT64_C( 5840623466093519866) },
      {  INT64_C(                   0), -INT64_C( 5840623466093519866) } },
    { UINT8_C( 92),
      { -INT64_C( 1022660964802028336),  INT64_C( 1942096119268911562) },
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(150),
      {  INT64_C( 2457538727575576513), -INT64_C( 3693955888196812037) },
      {  INT64_C(                   0), -INT64_C( 3693955888196812037) } },
    { UINT8_C(191),
      { -INT64_C( 5546376438535024113), -INT64_C( 6737103276578559796) },
      { -INT64_C( 5546376438535024113), -INT64_C( 6737103276578559796) } },
    { UINT8_C(233),
      {  INT64_C( 5286331197815199040), -INT64_C( 1474235251554841154) },
      {  INT64_C( 5286331197815199040),  INT64_C(                   0) } },
    { UINT8_C(252),
      {  INT64_C(  907503510357929832), -INT64_C( 7882435704522941479) },
      {  INT64_C(                   0),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_load_epi64(k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_load_epi64");
    easysimd_test_x86_assert_equal_i64x2(r, easysimd_mm_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i r = easysimd_mm_maskz_load_epi64(k, &a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_load_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[4];
    const uint8_t k;
    EASYSIMD_ALIGN_LIKE_64(easysimd__m128) const easysimd_float32 a[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -211.73), EASYSIMD_FLOAT32_C(  -515.26), EASYSIMD_FLOAT32_C(    47.03), EASYSIMD_FLOAT32_C(    54.75) },
      UINT8_C( 70),
      { EASYSIMD_FLOAT32_C(   663.70), EASYSIMD_FLOAT32_C(  -230.82), EASYSIMD_FLOAT32_C(    89.13), EASYSIMD_FLOAT32_C(   161.28) },
      { EASYSIMD_FLOAT32_C(  -211.73), EASYSIMD_FLOAT32_C(  -230.82), EASYSIMD_FLOAT32_C(    89.13), EASYSIMD_FLOAT32_C(    54.75) } },
    { { EASYSIMD_FLOAT32_C(   294.90), EASYSIMD_FLOAT32_C(   179.98), EASYSIMD_FLOAT32_C(  -114.21), EASYSIMD_FLOAT32_C(   186.44) },
      UINT8_C(120),
      { EASYSIMD_FLOAT32_C(  -897.80), EASYSIMD_FLOAT32_C(   194.33), EASYSIMD_FLOAT32_C(   764.51), EASYSIMD_FLOAT32_C(  -488.31) },
      { EASYSIMD_FLOAT32_C(   294.90), EASYSIMD_FLOAT32_C(   179.98), EASYSIMD_FLOAT32_C(  -114.21), EASYSIMD_FLOAT32_C(  -488.31) } },
    { { EASYSIMD_FLOAT32_C(  -349.51), EASYSIMD_FLOAT32_C(  -886.56), EASYSIMD_FLOAT32_C(   942.33), EASYSIMD_FLOAT32_C(  -426.98) },
      UINT8_C( 58),
      { EASYSIMD_FLOAT32_C(  -224.19), EASYSIMD_FLOAT32_C(  -392.44), EASYSIMD_FLOAT32_C(  -630.39), EASYSIMD_FLOAT32_C(   130.55) },
      { EASYSIMD_FLOAT32_C(  -349.51), EASYSIMD_FLOAT32_C(  -392.44), EASYSIMD_FLOAT32_C(   942.33), EASYSIMD_FLOAT32_C(   130.55) } },
    { { EASYSIMD_FLOAT32_C(   776.35), EASYSIMD_FLOAT32_C(   269.52), EASYSIMD_FLOAT32_C(   351.39), EASYSIMD_FLOAT32_C(  -126.67) },
      UINT8_C( 55),
      { EASYSIMD_FLOAT32_C(   836.13), EASYSIMD_FLOAT32_C(   920.36), EASYSIMD_FLOAT32_C(   112.54), EASYSIMD_FLOAT32_C(   167.45) },
      { EASYSIMD_FLOAT32_C(   836.13), EASYSIMD_FLOAT32_C(   920.36), EASYSIMD_FLOAT32_C(   112.54), EASYSIMD_FLOAT32_C(  -126.67) } },
    { { EASYSIMD_FLOAT32_C(   584.06), EASYSIMD_FLOAT32_C(   881.71), EASYSIMD_FLOAT32_C(  -743.42), EASYSIMD_FLOAT32_C(  -254.66) },
      UINT8_C(124),
      { EASYSIMD_FLOAT32_C(   436.56), EASYSIMD_FLOAT32_C(   631.14), EASYSIMD_FLOAT32_C(  -636.95), EASYSIMD_FLOAT32_C(  -168.69) },
      { EASYSIMD_FLOAT32_C(   584.06), EASYSIMD_FLOAT32_C(   881.71), EASYSIMD_FLOAT32_C(  -636.95), EASYSIMD_FLOAT32_C(  -168.69) } },
    { { EASYSIMD_FLOAT32_C(   733.34), EASYSIMD_FLOAT32_C(   557.38), EASYSIMD_FLOAT32_C(  -404.18), EASYSIMD_FLOAT32_C(  -754.98) },
      UINT8_C( 59),
      { EASYSIMD_FLOAT32_C(  -290.74), EASYSIMD_FLOAT32_C(  -812.65), EASYSIMD_FLOAT32_C(  -219.11), EASYSIMD_FLOAT32_C(   361.14) },
      { EASYSIMD_FLOAT32_C(  -290.74), EASYSIMD_FLOAT32_C(  -812.65), EASYSIMD_FLOAT32_C(  -404.18), EASYSIMD_FLOAT32_C(   361.14) } },
    { { EASYSIMD_FLOAT32_C(   -36.84), EASYSIMD_FLOAT32_C(   388.45), EASYSIMD_FLOAT32_C(   730.74), EASYSIMD_FLOAT32_C(  -906.30) },
      UINT8_C(195),
      { EASYSIMD_FLOAT32_C(     0.26), EASYSIMD_FLOAT32_C(   445.10), EASYSIMD_FLOAT32_C(  -961.88), EASYSIMD_FLOAT32_C(    58.06) },
      { EASYSIMD_FLOAT32_C(     0.26), EASYSIMD_FLOAT32_C(   445.10), EASYSIMD_FLOAT32_C(   730.74), EASYSIMD_FLOAT32_C(  -906.30) } },
    { { EASYSIMD_FLOAT32_C(   281.23), EASYSIMD_FLOAT32_C(   958.48), EASYSIMD_FLOAT32_C(  -829.41), EASYSIMD_FLOAT32_C(  -551.32) },
      UINT8_C(225),
      { EASYSIMD_FLOAT32_C(  -947.69), EASYSIMD_FLOAT32_C(  -294.74), EASYSIMD_FLOAT32_C(  -712.12), EASYSIMD_FLOAT32_C(   228.92) },
      { EASYSIMD_FLOAT32_C(  -947.69), EASYSIMD_FLOAT32_C(   958.48), EASYSIMD_FLOAT32_C(  -829.41), EASYSIMD_FLOAT32_C(  -551.32) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128 src = easysimd_mm_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_load_ps(src, k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_load_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128 src = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128 r = easysimd_mm_mask_load_ps(src, k, &a);

    easysimd_test_x86_write_f32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_load_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    EASYSIMD_ALIGN_LIKE_64(easysimd__m128) const easysimd_float32 a[4];
    const easysimd_float32 r[4];
  } test_vec[] = {
    { UINT8_C( 66),
      { EASYSIMD_FLOAT32_C(  -299.79), EASYSIMD_FLOAT32_C(  -789.56), EASYSIMD_FLOAT32_C(   598.06), EASYSIMD_FLOAT32_C(  -742.32) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -789.56), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(187),
      { EASYSIMD_FLOAT32_C(   193.19), EASYSIMD_FLOAT32_C(     9.58), EASYSIMD_FLOAT32_C(   362.47), EASYSIMD_FLOAT32_C(   375.94) },
      { EASYSIMD_FLOAT32_C(   193.19), EASYSIMD_FLOAT32_C(     9.58), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   375.94) } },
    { UINT8_C( 38),
      { EASYSIMD_FLOAT32_C(  -858.78), EASYSIMD_FLOAT32_C(   611.32), EASYSIMD_FLOAT32_C(  -112.76), EASYSIMD_FLOAT32_C(  -170.23) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   611.32), EASYSIMD_FLOAT32_C(  -112.76), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(175),
      { EASYSIMD_FLOAT32_C(   144.82), EASYSIMD_FLOAT32_C(  -428.75), EASYSIMD_FLOAT32_C(  -865.62), EASYSIMD_FLOAT32_C(   737.91) },
      { EASYSIMD_FLOAT32_C(   144.82), EASYSIMD_FLOAT32_C(  -428.75), EASYSIMD_FLOAT32_C(  -865.62), EASYSIMD_FLOAT32_C(   737.91) } },
    { UINT8_C(205),
      { EASYSIMD_FLOAT32_C(   887.48), EASYSIMD_FLOAT32_C(  -941.70), EASYSIMD_FLOAT32_C(   295.89), EASYSIMD_FLOAT32_C(   787.61) },
      { EASYSIMD_FLOAT32_C(   887.48), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   295.89), EASYSIMD_FLOAT32_C(   787.61) } },
    { UINT8_C( 76),
      { EASYSIMD_FLOAT32_C(   690.30), EASYSIMD_FLOAT32_C(  -542.97), EASYSIMD_FLOAT32_C(   940.93), EASYSIMD_FLOAT32_C(   221.22) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   940.93), EASYSIMD_FLOAT32_C(   221.22) } },
    { UINT8_C( 41),
      { EASYSIMD_FLOAT32_C(   877.46), EASYSIMD_FLOAT32_C(   921.43), EASYSIMD_FLOAT32_C(   623.41), EASYSIMD_FLOAT32_C(   475.52) },
      { EASYSIMD_FLOAT32_C(   877.46), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   475.52) } },
    { UINT8_C(138),
      { EASYSIMD_FLOAT32_C(   236.60), EASYSIMD_FLOAT32_C(  -331.30), EASYSIMD_FLOAT32_C(   188.69), EASYSIMD_FLOAT32_C(  -400.93) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -331.30), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -400.93) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_load_ps(k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_load_ps");
    easysimd_test_x86_assert_equal_f32x4(r, easysimd_mm_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128 a = easysimd_test_x86_random_f32x4(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128 r = easysimd_mm_maskz_load_ps(k, &a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_load_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[2];
    const uint8_t k;
    EASYSIMD_ALIGN_LIKE_64(easysimd__m128d) const easysimd_float64 a[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   145.21), EASYSIMD_FLOAT64_C(   375.07) },
      UINT8_C( 80),
      { EASYSIMD_FLOAT64_C(   105.63), EASYSIMD_FLOAT64_C(  -128.95) },
      { EASYSIMD_FLOAT64_C(   145.21), EASYSIMD_FLOAT64_C(   375.07) } },
    { { EASYSIMD_FLOAT64_C(   189.95), EASYSIMD_FLOAT64_C(  -713.01) },
      UINT8_C(205),
      { EASYSIMD_FLOAT64_C(  -556.88), EASYSIMD_FLOAT64_C(   876.94) },
      { EASYSIMD_FLOAT64_C(  -556.88), EASYSIMD_FLOAT64_C(  -713.01) } },
    { { EASYSIMD_FLOAT64_C(    99.74), EASYSIMD_FLOAT64_C(   771.37) },
      UINT8_C( 52),
      { EASYSIMD_FLOAT64_C(  -891.73), EASYSIMD_FLOAT64_C(  -282.80) },
      { EASYSIMD_FLOAT64_C(    99.74), EASYSIMD_FLOAT64_C(   771.37) } },
    { { EASYSIMD_FLOAT64_C(   895.41), EASYSIMD_FLOAT64_C(   -25.72) },
      UINT8_C(178),
      { EASYSIMD_FLOAT64_C(   875.36), EASYSIMD_FLOAT64_C(  -250.58) },
      { EASYSIMD_FLOAT64_C(   895.41), EASYSIMD_FLOAT64_C(  -250.58) } },
    { { EASYSIMD_FLOAT64_C(   -87.94), EASYSIMD_FLOAT64_C(  -230.69) },
      UINT8_C(218),
      { EASYSIMD_FLOAT64_C(  -270.12), EASYSIMD_FLOAT64_C(   641.86) },
      { EASYSIMD_FLOAT64_C(   -87.94), EASYSIMD_FLOAT64_C(   641.86) } },
    { { EASYSIMD_FLOAT64_C(   170.95), EASYSIMD_FLOAT64_C(   881.81) },
      UINT8_C(212),
      { EASYSIMD_FLOAT64_C(   585.41), EASYSIMD_FLOAT64_C(  -269.79) },
      { EASYSIMD_FLOAT64_C(   170.95), EASYSIMD_FLOAT64_C(   881.81) } },
    { { EASYSIMD_FLOAT64_C(   411.31), EASYSIMD_FLOAT64_C(  -269.39) },
      UINT8_C(175),
      { EASYSIMD_FLOAT64_C(   -54.85), EASYSIMD_FLOAT64_C(   836.25) },
      { EASYSIMD_FLOAT64_C(   -54.85), EASYSIMD_FLOAT64_C(   836.25) } },
    { { EASYSIMD_FLOAT64_C(   -23.67), EASYSIMD_FLOAT64_C(  -864.91) },
      UINT8_C( 51),
      { EASYSIMD_FLOAT64_C(  -144.69), EASYSIMD_FLOAT64_C(  -421.79) },
      { EASYSIMD_FLOAT64_C(  -144.69), EASYSIMD_FLOAT64_C(  -421.79) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128d src = easysimd_mm_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_mask_load_pd(src, k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_mask_load_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128d src = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128d r = easysimd_mm_mask_load_pd(src, k, &a);

    easysimd_test_x86_write_f64x2(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_load_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    EASYSIMD_ALIGN_LIKE_64(easysimd__m128d) const easysimd_float64 a[2];
    const easysimd_float64 r[2];
  } test_vec[] = {
    { UINT8_C(184),
      { EASYSIMD_FLOAT64_C(   469.03), EASYSIMD_FLOAT64_C(   399.23) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(  9),
      { EASYSIMD_FLOAT64_C(  -975.32), EASYSIMD_FLOAT64_C(   781.34) },
      { EASYSIMD_FLOAT64_C(  -975.32), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(164),
      { EASYSIMD_FLOAT64_C(   218.72), EASYSIMD_FLOAT64_C(   727.20) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(152),
      { EASYSIMD_FLOAT64_C(    88.94), EASYSIMD_FLOAT64_C(  -350.64) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(105),
      { EASYSIMD_FLOAT64_C(  -276.49), EASYSIMD_FLOAT64_C(  -541.46) },
      { EASYSIMD_FLOAT64_C(  -276.49), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C(150),
      { EASYSIMD_FLOAT64_C(   567.75), EASYSIMD_FLOAT64_C(   -26.13) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   -26.13) } },
    { UINT8_C(190),
      { EASYSIMD_FLOAT64_C(   669.58), EASYSIMD_FLOAT64_C(   107.61) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   107.61) } },
    { UINT8_C( 59),
      { EASYSIMD_FLOAT64_C(  -108.20), EASYSIMD_FLOAT64_C(  -606.19) },
      { EASYSIMD_FLOAT64_C(  -108.20), EASYSIMD_FLOAT64_C(  -606.19) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm_maskz_load_pd(k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm_maskz_load_pd");
    easysimd_test_x86_assert_equal_f64x2(r, easysimd_mm_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128d a = easysimd_test_x86_random_f64x2(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m128d r = easysimd_mm_maskz_load_pd(k, &a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_load_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[8];
    const uint8_t k;
    EASYSIMD_ALIGN_LIKE_64(easysimd__m256i) const int32_t a[8];
    const int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(  1189774471),  INT32_C(   457851800), -INT32_C(  1367511804), -INT32_C(  1155540122),  INT32_C(   663338470),  INT32_C(  1012935504), -INT32_C(   835912415), -INT32_C(  2120202232) },
      UINT8_C(195),
      { -INT32_C(   111396171), -INT32_C(   235047036), -INT32_C(   799560460), -INT32_C(  2001333557), -INT32_C(   958800229),  INT32_C(  1005065534), -INT32_C(  1992051135),  INT32_C(   206357590) },
      { -INT32_C(   111396171), -INT32_C(   235047036), -INT32_C(  1367511804), -INT32_C(  1155540122),  INT32_C(   663338470),  INT32_C(  1012935504), -INT32_C(  1992051135),  INT32_C(   206357590) } },
    { { -INT32_C(  2096781057),  INT32_C(   343212575),  INT32_C(  2045037742),  INT32_C(  2046991071), -INT32_C(  1237197960),  INT32_C(   837888496),  INT32_C(   918172895), -INT32_C(   129890567) },
      UINT8_C(175),
      {  INT32_C(  1238268743), -INT32_C(  1107762448),  INT32_C(  1637642694),  INT32_C(  1322849907), -INT32_C(  2143383465), -INT32_C(  1235193983),  INT32_C(   816813609),  INT32_C(   534751192) },
      {  INT32_C(  1238268743), -INT32_C(  1107762448),  INT32_C(  1637642694),  INT32_C(  1322849907), -INT32_C(  1237197960), -INT32_C(  1235193983),  INT32_C(   918172895),  INT32_C(   534751192) } },
    { {  INT32_C(   325627170),  INT32_C(  1456496784),  INT32_C(  1169648850), -INT32_C(   627797630), -INT32_C(  1571106271),  INT32_C(  1800976962),  INT32_C(   681248592), -INT32_C(   783844690) },
      UINT8_C( 40),
      {  INT32_C(   280552624),  INT32_C(   551685812),  INT32_C(  1470245062), -INT32_C(  1904706372), -INT32_C(  1848632617),  INT32_C(  2061646963),  INT32_C(  1378421463),  INT32_C(    41613906) },
      {  INT32_C(   325627170),  INT32_C(  1456496784),  INT32_C(  1169648850), -INT32_C(  1904706372), -INT32_C(  1571106271),  INT32_C(  2061646963),  INT32_C(   681248592), -INT32_C(   783844690) } },
    { { -INT32_C(  1844301090),  INT32_C(   112391488), -INT32_C(   648195043), -INT32_C(  1469590063),  INT32_C(  1664694512),  INT32_C(  1272781684),  INT32_C(  2023556646), -INT32_C(   562424064) },
      UINT8_C( 73),
      { -INT32_C(  2121633652),  INT32_C(  2006880290), -INT32_C(  1018660882),  INT32_C(   397668575),  INT32_C(  1166743081), -INT32_C(    93595916), -INT32_C(  1946426508), -INT32_C(   371861155) },
      { -INT32_C(  2121633652),  INT32_C(   112391488), -INT32_C(   648195043),  INT32_C(   397668575),  INT32_C(  1664694512),  INT32_C(  1272781684), -INT32_C(  1946426508), -INT32_C(   562424064) } },
    { {  INT32_C(  1818976074), -INT32_C(   572323345),  INT32_C(  1621175169),  INT32_C(  1165448219),  INT32_C(  1602880107),  INT32_C(  1280964056),  INT32_C(   903369944),  INT32_C(  2015276334) },
      UINT8_C( 12),
      { -INT32_C(  1812208503), -INT32_C(   216737336), -INT32_C(   837847942), -INT32_C(   315009813), -INT32_C(   742024994),  INT32_C(  1185616626), -INT32_C(  1753947926), -INT32_C(  2002522882) },
      {  INT32_C(  1818976074), -INT32_C(   572323345), -INT32_C(   837847942), -INT32_C(   315009813),  INT32_C(  1602880107),  INT32_C(  1280964056),  INT32_C(   903369944),  INT32_C(  2015276334) } },
    { { -INT32_C(  1726243119), -INT32_C(   242471049), -INT32_C(  1899979869), -INT32_C(   847513105), -INT32_C(  2069872494),  INT32_C(  1019890514),  INT32_C(   718487339), -INT32_C(    55413205) },
      UINT8_C( 21),
      { -INT32_C(    57895475), -INT32_C(  1113620958),  INT32_C(   934096190), -INT32_C(   389449048),  INT32_C(  1715162395),  INT32_C(  1469216536), -INT32_C(  1048396725),  INT32_C(  1003913070) },
      { -INT32_C(    57895475), -INT32_C(   242471049),  INT32_C(   934096190), -INT32_C(   847513105),  INT32_C(  1715162395),  INT32_C(  1019890514),  INT32_C(   718487339), -INT32_C(    55413205) } },
    { {  INT32_C(   909599508),  INT32_C(   536139489), -INT32_C(  1420386045),  INT32_C(   915611675), -INT32_C(  2053255571), -INT32_C(  1847840954),  INT32_C(  1498570731), -INT32_C(   225170978) },
      UINT8_C(140),
      { -INT32_C(  1586681397), -INT32_C(  1096512483),  INT32_C(    47796194), -INT32_C(  1318055710), -INT32_C(   587729491),  INT32_C(   835160274),  INT32_C(    51323098),  INT32_C(  2140078516) },
      {  INT32_C(   909599508),  INT32_C(   536139489),  INT32_C(    47796194), -INT32_C(  1318055710), -INT32_C(  2053255571), -INT32_C(  1847840954),  INT32_C(  1498570731),  INT32_C(  2140078516) } },
    { {  INT32_C(  1193344042),  INT32_C(  1795540104), -INT32_C(   177348845), -INT32_C(  1666785809), -INT32_C(  1518821933),  INT32_C(    14040869),  INT32_C(   319022431),  INT32_C(   294818790) },
      UINT8_C(143),
      {  INT32_C(  1981307058),  INT32_C(  1032422238), -INT32_C(   835944720), -INT32_C(  1029584859),  INT32_C(  2145928768),  INT32_C(    31385628),  INT32_C(  2129129963),  INT32_C(   890173571) },
      {  INT32_C(  1981307058),  INT32_C(  1032422238), -INT32_C(   835944720), -INT32_C(  1029584859), -INT32_C(  1518821933),  INT32_C(    14040869),  INT32_C(   319022431),  INT32_C(   890173571) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_load_epi32(src, k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_load_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_mask_load_epi32(src, k, &a);

    easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_load_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    EASYSIMD_ALIGN_LIKE_64(easysimd__m256i) const int32_t a[8];
    const int32_t r[8];
  } test_vec[] = {
    { UINT8_C(228),
      {  INT32_C(  1626298356), -INT32_C(  1876654857),  INT32_C(   738963719), -INT32_C(  1447234251),  INT32_C(   981956140), -INT32_C(   735991281),  INT32_C(   642180195), -INT32_C(   821302309) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(   738963719),  INT32_C(           0),  INT32_C(           0), -INT32_C(   735991281),  INT32_C(   642180195), -INT32_C(   821302309) } },
    { UINT8_C( 66),
      {  INT32_C(  2117677050),  INT32_C(    92653907), -INT32_C(   902123051), -INT32_C(   470358931),  INT32_C(   368259179),  INT32_C(   897107793), -INT32_C(   133128435), -INT32_C(  1539645526) },
      {  INT32_C(           0),  INT32_C(    92653907),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   133128435),  INT32_C(           0) } },
    { UINT8_C( 15),
      {  INT32_C(  1013064307),  INT32_C(  1477535655), -INT32_C(  2033853278),  INT32_C(    66169042),  INT32_C(  1649673883),  INT32_C(   510626431),  INT32_C(  2059954074),  INT32_C(   344550561) },
      {  INT32_C(  1013064307),  INT32_C(  1477535655), -INT32_C(  2033853278),  INT32_C(    66169042),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(142),
      {  INT32_C(  1396003307),  INT32_C(  1056279906), -INT32_C(    99517614),  INT32_C(  1922438252), -INT32_C(   219023256), -INT32_C(   812839064),  INT32_C(  1131481047),  INT32_C(  2077328784) },
      {  INT32_C(           0),  INT32_C(  1056279906), -INT32_C(    99517614),  INT32_C(  1922438252),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  2077328784) } },
    { UINT8_C(214),
      { -INT32_C(  1824928250),  INT32_C(  1055225795), -INT32_C(  1666523000),  INT32_C(  1845828982),  INT32_C(   517404430), -INT32_C(  1946835324), -INT32_C(  1675937513),  INT32_C(   292787978) },
      {  INT32_C(           0),  INT32_C(  1055225795), -INT32_C(  1666523000),  INT32_C(           0),  INT32_C(   517404430),  INT32_C(           0), -INT32_C(  1675937513),  INT32_C(   292787978) } },
    { UINT8_C(101),
      {  INT32_C(   589931692),  INT32_C(  1789683594),  INT32_C(   803227666),  INT32_C(  1161645645), -INT32_C(   875996379), -INT32_C(  1964878511),  INT32_C(   127172208), -INT32_C(  1653758479) },
      {  INT32_C(   589931692),  INT32_C(           0),  INT32_C(   803227666),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1964878511),  INT32_C(   127172208),  INT32_C(           0) } },
    { UINT8_C( 74),
      { -INT32_C(    36388458), -INT32_C(  1257292179),  INT32_C(  1812151838), -INT32_C(   678344580),  INT32_C(  1713921041), -INT32_C(  1109937602), -INT32_C(   324084153), -INT32_C(   516535221) },
      {  INT32_C(           0), -INT32_C(  1257292179),  INT32_C(           0), -INT32_C(   678344580),  INT32_C(           0),  INT32_C(           0), -INT32_C(   324084153),  INT32_C(           0) } },
    { UINT8_C( 13),
      {  INT32_C(  1216012042),  INT32_C(   761671662),  INT32_C(  2057950002), -INT32_C(  1047756700),  INT32_C(  1543500457), -INT32_C(  1465729847), -INT32_C(  1208774805), -INT32_C(   792406587) },
      {  INT32_C(  1216012042),  INT32_C(           0),  INT32_C(  2057950002), -INT32_C(  1047756700),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_load_epi32(k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_load_epi32");
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_maskz_load_epi32(k, &a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_load_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t src[4];
    const uint8_t k;
    EASYSIMD_ALIGN_LIKE_64(easysimd__m256i) const int64_t a[4];
    const int64_t r[4];
  } test_vec[] = {
    { { -INT64_C( 7355446188001515951),  INT64_C( 2349199917406689458),  INT64_C( 6155151211740758606), -INT64_C( 7194814538870140050) },
      UINT8_C(  8),
      {  INT64_C( 5744141449913519057), -INT64_C( 6122275177797546062),  INT64_C( 3664077279649428462), -INT64_C( 4528339806681445826) },
      { -INT64_C( 7355446188001515951),  INT64_C( 2349199917406689458),  INT64_C( 6155151211740758606), -INT64_C( 4528339806681445826) } },
    { { -INT64_C( 2959852159358281371),  INT64_C( 6548684018929376269), -INT64_C( 6980967133739616755),  INT64_C( 1953410729004473193) },
      UINT8_C(193),
      { -INT64_C( 4580798931291227934),  INT64_C( 8859291322329899852), -INT64_C( 7470211754721878453), -INT64_C( 6061166935430812103) },
      { -INT64_C( 4580798931291227934),  INT64_C( 6548684018929376269), -INT64_C( 6980967133739616755),  INT64_C( 1953410729004473193) } },
    { { -INT64_C( 3493253428614019040), -INT64_C( 8999978522978701029), -INT64_C( 6477652437935197050), -INT64_C( 1016499804579600273) },
      UINT8_C(114),
      { -INT64_C( 1331722730840148240),  INT64_C( 3578012309960008570), -INT64_C( 6033058488855418347),  INT64_C( 5992160678758940428) },
      { -INT64_C( 3493253428614019040),  INT64_C( 3578012309960008570), -INT64_C( 6477652437935197050), -INT64_C( 1016499804579600273) } },
    { {  INT64_C( 6722819689065225764),  INT64_C( 3301461889052995939), -INT64_C( 4521250407170412464),  INT64_C( 5800812431638314697) },
      UINT8_C(190),
      {  INT64_C( 7917327030316805948), -INT64_C( 4471032841831612291), -INT64_C( 8734106095662495817), -INT64_C( 5616267503124416848) },
      {  INT64_C( 6722819689065225764), -INT64_C( 4471032841831612291), -INT64_C( 8734106095662495817), -INT64_C( 5616267503124416848) } },
    { {  INT64_C( 3178150109744378034),  INT64_C(  766107077122695310), -INT64_C(   29038861996058807), -INT64_C(    2991439314138164) },
      UINT8_C( 15),
      {  INT64_C( 7620067240083107107),  INT64_C( 6324680331479523257), -INT64_C( 5011197486855366702), -INT64_C( 3502384702763256074) },
      {  INT64_C( 7620067240083107107),  INT64_C( 6324680331479523257), -INT64_C( 5011197486855366702), -INT64_C( 3502384702763256074) } },
    { { -INT64_C( 4042332447059205195),  INT64_C( 4392707740578063199),  INT64_C( 4181970779394842079), -INT64_C( 3636105737566940913) },
      UINT8_C( 24),
      { -INT64_C( 3122702971139896438), -INT64_C( 2587182478745778127), -INT64_C( 9167259080612354465), -INT64_C( 7640244566651993622) },
      { -INT64_C( 4042332447059205195),  INT64_C( 4392707740578063199),  INT64_C( 4181970779394842079), -INT64_C( 7640244566651993622) } },
    { {  INT64_C( 4447172745995689649), -INT64_C( 3392742753045121328),  INT64_C( 5757094995559975669), -INT64_C( 7606254431751930159) },
      UINT8_C( 69),
      {  INT64_C( 2785633809479137616), -INT64_C( 6086218378222520826),  INT64_C( 5841556652105461488), -INT64_C( 5837848884311364209) },
      {  INT64_C( 2785633809479137616), -INT64_C( 3392742753045121328),  INT64_C( 5841556652105461488), -INT64_C( 7606254431751930159) } },
    { { -INT64_C( 2045990077015372476), -INT64_C( 8416944461108005916),  INT64_C( 3464355125693989359),  INT64_C( 6401271021573654729) },
      UINT8_C( 47),
      { -INT64_C( 7174533919804699301),  INT64_C( 7374576376465584999),  INT64_C( 8373379370597161358), -INT64_C( 8344922508938945762) },
      { -INT64_C( 7174533919804699301),  INT64_C( 7374576376465584999),  INT64_C( 8373379370597161358), -INT64_C( 8344922508938945762) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_mm256_loadu_epi64(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_load_epi64(src, k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_load_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i64x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_mask_load_epi64(src, k, &a);

    easysimd_test_x86_write_i64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_load_epi64 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    EASYSIMD_ALIGN_LIKE_64(easysimd__m256i) const int64_t a[4];
    const int64_t r[4];
  } test_vec[] = {
    { UINT8_C(180),
      {  INT64_C( 5953986853590931518), -INT64_C( 7376201680641713544),  INT64_C( 3431191649276067191),  INT64_C( 6307225418364900862) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 3431191649276067191),  INT64_C(                   0) } },
    { UINT8_C(221),
      {  INT64_C( 9183312868183688489),  INT64_C(  744148425400281700),  INT64_C( 8803087175335446049), -INT64_C( 5949277628105687603) },
      {  INT64_C( 9183312868183688489),  INT64_C(                   0),  INT64_C( 8803087175335446049), -INT64_C( 5949277628105687603) } },
    { UINT8_C(176),
      { -INT64_C( 9016886946582649141), -INT64_C( 7810480849869096517), -INT64_C( 1231397011958741181), -INT64_C( 5376698812738667578) },
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(198),
      { -INT64_C( 4194996348495450504), -INT64_C( 6499708160028032272),  INT64_C( 5554774324438034461), -INT64_C( 6710665629701519113) },
      {  INT64_C(                   0), -INT64_C( 6499708160028032272),  INT64_C( 5554774324438034461),  INT64_C(                   0) } },
    { UINT8_C( 27),
      { -INT64_C( 5112168906149613636),  INT64_C( 5350624224511235775),  INT64_C( 7948688030479192274),  INT64_C(  119692030721484448) },
      { -INT64_C( 5112168906149613636),  INT64_C( 5350624224511235775),  INT64_C(                   0),  INT64_C(  119692030721484448) } },
    { UINT8_C(235),
      { -INT64_C( 4432636954757947696),  INT64_C( 8269013406999486643), -INT64_C( 3560791045524940405), -INT64_C( 7186033291784150496) },
      { -INT64_C( 4432636954757947696),  INT64_C( 8269013406999486643),  INT64_C(                   0), -INT64_C( 7186033291784150496) } },
    { UINT8_C( 50),
      {  INT64_C( 8536610299998788009),  INT64_C( 4702261762141480774), -INT64_C( 1330399404159846942), -INT64_C( 3111613321196280437) },
      {  INT64_C(                   0),  INT64_C( 4702261762141480774),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C(174),
      {  INT64_C( 8689688958323693872), -INT64_C( 4323488021553787822), -INT64_C( 2913975254008371060),  INT64_C( 3401862586029977139) },
      {  INT64_C(                   0), -INT64_C( 4323488021553787822), -INT64_C( 2913975254008371060),  INT64_C( 3401862586029977139) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_load_epi64(k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_load_epi64");
    easysimd_test_x86_assert_equal_i64x4(r, easysimd_mm256_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i r = easysimd_mm256_maskz_load_epi64(k, &a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_load_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[8];
    const uint8_t k;
    EASYSIMD_ALIGN_LIKE_64(easysimd__m256) const easysimd_float32 a[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(  -688.40), EASYSIMD_FLOAT32_C(  -249.11), EASYSIMD_FLOAT32_C(  -344.82), EASYSIMD_FLOAT32_C(   741.04),
        EASYSIMD_FLOAT32_C(  -952.90), EASYSIMD_FLOAT32_C(  -814.23), EASYSIMD_FLOAT32_C(   565.16), EASYSIMD_FLOAT32_C(  -394.04) },
      UINT8_C(100),
      { EASYSIMD_FLOAT32_C(   829.85), EASYSIMD_FLOAT32_C(  -794.15), EASYSIMD_FLOAT32_C(  -547.05), EASYSIMD_FLOAT32_C(  -357.63),
        EASYSIMD_FLOAT32_C(   502.53), EASYSIMD_FLOAT32_C(  -962.28), EASYSIMD_FLOAT32_C(  -300.22), EASYSIMD_FLOAT32_C(   205.81) },
      { EASYSIMD_FLOAT32_C(  -688.40), EASYSIMD_FLOAT32_C(  -249.11), EASYSIMD_FLOAT32_C(  -547.05), EASYSIMD_FLOAT32_C(   741.04),
        EASYSIMD_FLOAT32_C(  -952.90), EASYSIMD_FLOAT32_C(  -962.28), EASYSIMD_FLOAT32_C(  -300.22), EASYSIMD_FLOAT32_C(  -394.04) } },
    { { EASYSIMD_FLOAT32_C(  -100.09), EASYSIMD_FLOAT32_C(   492.96), EASYSIMD_FLOAT32_C(    83.82), EASYSIMD_FLOAT32_C(  -317.02),
        EASYSIMD_FLOAT32_C(    -4.64), EASYSIMD_FLOAT32_C(  -849.84), EASYSIMD_FLOAT32_C(   898.86), EASYSIMD_FLOAT32_C(   390.05) },
      UINT8_C(141),
      { EASYSIMD_FLOAT32_C(    67.37), EASYSIMD_FLOAT32_C(  -759.32), EASYSIMD_FLOAT32_C(  -217.33), EASYSIMD_FLOAT32_C(   217.66),
        EASYSIMD_FLOAT32_C(   475.88), EASYSIMD_FLOAT32_C(    94.27), EASYSIMD_FLOAT32_C(   968.55), EASYSIMD_FLOAT32_C(  -868.94) },
      { EASYSIMD_FLOAT32_C(    67.37), EASYSIMD_FLOAT32_C(   492.96), EASYSIMD_FLOAT32_C(  -217.33), EASYSIMD_FLOAT32_C(   217.66),
        EASYSIMD_FLOAT32_C(    -4.64), EASYSIMD_FLOAT32_C(  -849.84), EASYSIMD_FLOAT32_C(   898.86), EASYSIMD_FLOAT32_C(  -868.94) } },
    { { EASYSIMD_FLOAT32_C(  -164.69), EASYSIMD_FLOAT32_C(  -984.35), EASYSIMD_FLOAT32_C(  -683.17), EASYSIMD_FLOAT32_C(  -599.53),
        EASYSIMD_FLOAT32_C(  -378.39), EASYSIMD_FLOAT32_C(   554.89), EASYSIMD_FLOAT32_C(  -769.67), EASYSIMD_FLOAT32_C(  -172.54) },
      UINT8_C( 87),
      { EASYSIMD_FLOAT32_C(  -127.31), EASYSIMD_FLOAT32_C(  -670.01), EASYSIMD_FLOAT32_C(  -954.44), EASYSIMD_FLOAT32_C(   572.48),
        EASYSIMD_FLOAT32_C(   535.80), EASYSIMD_FLOAT32_C(   -54.53), EASYSIMD_FLOAT32_C(    65.43), EASYSIMD_FLOAT32_C(  -380.38) },
      { EASYSIMD_FLOAT32_C(  -127.31), EASYSIMD_FLOAT32_C(  -670.01), EASYSIMD_FLOAT32_C(  -954.44), EASYSIMD_FLOAT32_C(  -599.53),
        EASYSIMD_FLOAT32_C(   535.80), EASYSIMD_FLOAT32_C(   554.89), EASYSIMD_FLOAT32_C(    65.43), EASYSIMD_FLOAT32_C(  -172.54) } },
    { { EASYSIMD_FLOAT32_C(   628.45), EASYSIMD_FLOAT32_C(  -939.20), EASYSIMD_FLOAT32_C(  -230.22), EASYSIMD_FLOAT32_C(   527.32),
        EASYSIMD_FLOAT32_C(   450.84), EASYSIMD_FLOAT32_C(  -647.21), EASYSIMD_FLOAT32_C(  -405.31), EASYSIMD_FLOAT32_C(   691.52) },
      UINT8_C( 49),
      { EASYSIMD_FLOAT32_C(   812.35), EASYSIMD_FLOAT32_C(   167.40), EASYSIMD_FLOAT32_C(  -770.27), EASYSIMD_FLOAT32_C(   780.91),
        EASYSIMD_FLOAT32_C(   298.46), EASYSIMD_FLOAT32_C(    65.04), EASYSIMD_FLOAT32_C(   796.56), EASYSIMD_FLOAT32_C(   615.29) },
      { EASYSIMD_FLOAT32_C(   812.35), EASYSIMD_FLOAT32_C(  -939.20), EASYSIMD_FLOAT32_C(  -230.22), EASYSIMD_FLOAT32_C(   527.32),
        EASYSIMD_FLOAT32_C(   298.46), EASYSIMD_FLOAT32_C(    65.04), EASYSIMD_FLOAT32_C(  -405.31), EASYSIMD_FLOAT32_C(   691.52) } },
    { { EASYSIMD_FLOAT32_C(   465.51), EASYSIMD_FLOAT32_C(  -581.83), EASYSIMD_FLOAT32_C(   170.18), EASYSIMD_FLOAT32_C(   695.84),
        EASYSIMD_FLOAT32_C(   245.63), EASYSIMD_FLOAT32_C(   178.02), EASYSIMD_FLOAT32_C(  -431.47), EASYSIMD_FLOAT32_C(   575.61) },
      UINT8_C(203),
      { EASYSIMD_FLOAT32_C(  -859.00), EASYSIMD_FLOAT32_C(   111.41), EASYSIMD_FLOAT32_C(  -830.94), EASYSIMD_FLOAT32_C(   206.44),
        EASYSIMD_FLOAT32_C(   731.03), EASYSIMD_FLOAT32_C(   797.51), EASYSIMD_FLOAT32_C(   267.24), EASYSIMD_FLOAT32_C(  -499.19) },
      { EASYSIMD_FLOAT32_C(  -859.00), EASYSIMD_FLOAT32_C(   111.41), EASYSIMD_FLOAT32_C(   170.18), EASYSIMD_FLOAT32_C(   206.44),
        EASYSIMD_FLOAT32_C(   245.63), EASYSIMD_FLOAT32_C(   178.02), EASYSIMD_FLOAT32_C(   267.24), EASYSIMD_FLOAT32_C(  -499.19) } },
    { { EASYSIMD_FLOAT32_C(   324.83), EASYSIMD_FLOAT32_C(  -281.92), EASYSIMD_FLOAT32_C(  -146.40), EASYSIMD_FLOAT32_C(   919.52),
        EASYSIMD_FLOAT32_C(  -590.40), EASYSIMD_FLOAT32_C(   989.05), EASYSIMD_FLOAT32_C(   731.87), EASYSIMD_FLOAT32_C(   577.00) },
      UINT8_C(152),
      { EASYSIMD_FLOAT32_C(   512.78), EASYSIMD_FLOAT32_C(  -124.54), EASYSIMD_FLOAT32_C(   283.82), EASYSIMD_FLOAT32_C(   309.34),
        EASYSIMD_FLOAT32_C(  -509.25), EASYSIMD_FLOAT32_C(  -250.67), EASYSIMD_FLOAT32_C(   727.50), EASYSIMD_FLOAT32_C(   660.93) },
      { EASYSIMD_FLOAT32_C(   324.83), EASYSIMD_FLOAT32_C(  -281.92), EASYSIMD_FLOAT32_C(  -146.40), EASYSIMD_FLOAT32_C(   309.34),
        EASYSIMD_FLOAT32_C(  -509.25), EASYSIMD_FLOAT32_C(   989.05), EASYSIMD_FLOAT32_C(   731.87), EASYSIMD_FLOAT32_C(   660.93) } },
    { { EASYSIMD_FLOAT32_C(  -554.84), EASYSIMD_FLOAT32_C(   -26.87), EASYSIMD_FLOAT32_C(  -161.05), EASYSIMD_FLOAT32_C(    13.69),
        EASYSIMD_FLOAT32_C(  -451.26), EASYSIMD_FLOAT32_C(  -937.46), EASYSIMD_FLOAT32_C(   154.69), EASYSIMD_FLOAT32_C(   660.16) },
      UINT8_C(214),
      { EASYSIMD_FLOAT32_C(  -638.87), EASYSIMD_FLOAT32_C(   391.19), EASYSIMD_FLOAT32_C(  -970.89), EASYSIMD_FLOAT32_C(   628.37),
        EASYSIMD_FLOAT32_C(   892.00), EASYSIMD_FLOAT32_C(   353.93), EASYSIMD_FLOAT32_C(  -653.55), EASYSIMD_FLOAT32_C(  -254.40) },
      { EASYSIMD_FLOAT32_C(  -554.84), EASYSIMD_FLOAT32_C(   391.19), EASYSIMD_FLOAT32_C(  -970.89), EASYSIMD_FLOAT32_C(    13.69),
        EASYSIMD_FLOAT32_C(   892.00), EASYSIMD_FLOAT32_C(  -937.46), EASYSIMD_FLOAT32_C(  -653.55), EASYSIMD_FLOAT32_C(  -254.40) } },
    { { EASYSIMD_FLOAT32_C(   273.45), EASYSIMD_FLOAT32_C(  -243.95), EASYSIMD_FLOAT32_C(  -265.35), EASYSIMD_FLOAT32_C(     5.32),
        EASYSIMD_FLOAT32_C(  -666.95), EASYSIMD_FLOAT32_C(   -46.57), EASYSIMD_FLOAT32_C(  -481.90), EASYSIMD_FLOAT32_C(   208.51) },
      UINT8_C( 92),
      { EASYSIMD_FLOAT32_C(   827.44), EASYSIMD_FLOAT32_C(   699.26), EASYSIMD_FLOAT32_C(   -13.43), EASYSIMD_FLOAT32_C(   554.94),
        EASYSIMD_FLOAT32_C(   360.19), EASYSIMD_FLOAT32_C(   431.73), EASYSIMD_FLOAT32_C(  -471.93), EASYSIMD_FLOAT32_C(  -800.86) },
      { EASYSIMD_FLOAT32_C(   273.45), EASYSIMD_FLOAT32_C(  -243.95), EASYSIMD_FLOAT32_C(   -13.43), EASYSIMD_FLOAT32_C(   554.94),
        EASYSIMD_FLOAT32_C(   360.19), EASYSIMD_FLOAT32_C(   -46.57), EASYSIMD_FLOAT32_C(  -471.93), EASYSIMD_FLOAT32_C(   208.51) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256 src = easysimd_mm256_loadu_ps(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_load_ps(src, k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_load_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256 src = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_mask_load_ps(src, k, &a);

    easysimd_test_x86_write_f32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_load_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    EASYSIMD_ALIGN_LIKE_64(easysimd__m256) const easysimd_float32 a[8];
    const easysimd_float32 r[8];
  } test_vec[] = {
    { UINT8_C( 17),
      { EASYSIMD_FLOAT32_C(   309.46), EASYSIMD_FLOAT32_C(  -293.34), EASYSIMD_FLOAT32_C(  -143.11), EASYSIMD_FLOAT32_C(  -967.50),
        EASYSIMD_FLOAT32_C(  -999.73), EASYSIMD_FLOAT32_C(    71.94), EASYSIMD_FLOAT32_C(   464.25), EASYSIMD_FLOAT32_C(   662.41) },
      { EASYSIMD_FLOAT32_C(   309.46), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  -999.73), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(235),
      { EASYSIMD_FLOAT32_C(   853.80), EASYSIMD_FLOAT32_C(  -626.70), EASYSIMD_FLOAT32_C(  -591.97), EASYSIMD_FLOAT32_C(  -327.38),
        EASYSIMD_FLOAT32_C(   520.97), EASYSIMD_FLOAT32_C(  -832.66), EASYSIMD_FLOAT32_C(  -241.98), EASYSIMD_FLOAT32_C(  -882.25) },
      { EASYSIMD_FLOAT32_C(   853.80), EASYSIMD_FLOAT32_C(  -626.70), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -327.38),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -832.66), EASYSIMD_FLOAT32_C(  -241.98), EASYSIMD_FLOAT32_C(  -882.25) } },
    { UINT8_C(229),
      { EASYSIMD_FLOAT32_C(  -118.15), EASYSIMD_FLOAT32_C(  -308.90), EASYSIMD_FLOAT32_C(  -975.26), EASYSIMD_FLOAT32_C(  -735.50),
        EASYSIMD_FLOAT32_C(   776.22), EASYSIMD_FLOAT32_C(   879.40), EASYSIMD_FLOAT32_C(   880.02), EASYSIMD_FLOAT32_C(   213.13) },
      { EASYSIMD_FLOAT32_C(  -118.15), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -975.26), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   879.40), EASYSIMD_FLOAT32_C(   880.02), EASYSIMD_FLOAT32_C(   213.13) } },
    { UINT8_C(226),
      { EASYSIMD_FLOAT32_C(   921.86), EASYSIMD_FLOAT32_C(    96.34), EASYSIMD_FLOAT32_C(  -857.47), EASYSIMD_FLOAT32_C(   628.83),
        EASYSIMD_FLOAT32_C(  -594.20), EASYSIMD_FLOAT32_C(  -150.81), EASYSIMD_FLOAT32_C(  -514.28), EASYSIMD_FLOAT32_C(  -561.70) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    96.34), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -150.81), EASYSIMD_FLOAT32_C(  -514.28), EASYSIMD_FLOAT32_C(  -561.70) } },
    { UINT8_C(191),
      { EASYSIMD_FLOAT32_C(   557.66), EASYSIMD_FLOAT32_C(   902.54), EASYSIMD_FLOAT32_C(  -488.13), EASYSIMD_FLOAT32_C(   381.63),
        EASYSIMD_FLOAT32_C(   756.34), EASYSIMD_FLOAT32_C(  -114.84), EASYSIMD_FLOAT32_C(   789.66), EASYSIMD_FLOAT32_C(  -571.04) },
      { EASYSIMD_FLOAT32_C(   557.66), EASYSIMD_FLOAT32_C(   902.54), EASYSIMD_FLOAT32_C(  -488.13), EASYSIMD_FLOAT32_C(   381.63),
        EASYSIMD_FLOAT32_C(   756.34), EASYSIMD_FLOAT32_C(  -114.84), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -571.04) } },
    { UINT8_C( 46),
      { EASYSIMD_FLOAT32_C(   956.99), EASYSIMD_FLOAT32_C(   186.98), EASYSIMD_FLOAT32_C(  -476.12), EASYSIMD_FLOAT32_C(  -499.74),
        EASYSIMD_FLOAT32_C(  -931.17), EASYSIMD_FLOAT32_C(   214.98), EASYSIMD_FLOAT32_C(  -475.00), EASYSIMD_FLOAT32_C(  -666.67) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   186.98), EASYSIMD_FLOAT32_C(  -476.12), EASYSIMD_FLOAT32_C(  -499.74),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   214.98), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT8_C(205),
      { EASYSIMD_FLOAT32_C(  -595.60), EASYSIMD_FLOAT32_C(  -786.65), EASYSIMD_FLOAT32_C(  -795.67), EASYSIMD_FLOAT32_C(   898.46),
        EASYSIMD_FLOAT32_C(  -864.78), EASYSIMD_FLOAT32_C(   300.66), EASYSIMD_FLOAT32_C(  -959.01), EASYSIMD_FLOAT32_C(   764.05) },
      { EASYSIMD_FLOAT32_C(  -595.60), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -795.67), EASYSIMD_FLOAT32_C(   898.46),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -959.01), EASYSIMD_FLOAT32_C(   764.05) } },
    { UINT8_C(238),
      { EASYSIMD_FLOAT32_C(  -109.82), EASYSIMD_FLOAT32_C(  -750.23), EASYSIMD_FLOAT32_C(  -855.24), EASYSIMD_FLOAT32_C(   739.64),
        EASYSIMD_FLOAT32_C(   807.43), EASYSIMD_FLOAT32_C(  -952.70), EASYSIMD_FLOAT32_C(  -748.49), EASYSIMD_FLOAT32_C(   189.05) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -750.23), EASYSIMD_FLOAT32_C(  -855.24), EASYSIMD_FLOAT32_C(   739.64),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -952.70), EASYSIMD_FLOAT32_C(  -748.49), EASYSIMD_FLOAT32_C(   189.05) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_load_ps(k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_load_ps");
    easysimd_test_x86_assert_equal_f32x8(r, easysimd_mm256_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256 a = easysimd_test_x86_random_f32x8(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m256 r = easysimd_mm256_maskz_load_ps(k, &a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_load_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float64 src[4];
    const uint8_t k;
    EASYSIMD_ALIGN_LIKE_64(easysimd__m256d) const easysimd_float64 a[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(  -649.05), EASYSIMD_FLOAT64_C(  -323.82), EASYSIMD_FLOAT64_C(   991.19), EASYSIMD_FLOAT64_C(  -883.58) },
      UINT8_C(194),
      { EASYSIMD_FLOAT64_C(   683.46), EASYSIMD_FLOAT64_C(  -798.17), EASYSIMD_FLOAT64_C(   322.45), EASYSIMD_FLOAT64_C(  -139.52) },
      { EASYSIMD_FLOAT64_C(  -649.05), EASYSIMD_FLOAT64_C(  -798.17), EASYSIMD_FLOAT64_C(   991.19), EASYSIMD_FLOAT64_C(  -883.58) } },
    { { EASYSIMD_FLOAT64_C(   -78.74), EASYSIMD_FLOAT64_C(  -955.93), EASYSIMD_FLOAT64_C(  -801.28), EASYSIMD_FLOAT64_C(    58.35) },
      UINT8_C( 81),
      { EASYSIMD_FLOAT64_C(   740.79), EASYSIMD_FLOAT64_C(  -980.35), EASYSIMD_FLOAT64_C(  -170.84), EASYSIMD_FLOAT64_C(   824.00) },
      { EASYSIMD_FLOAT64_C(   740.79), EASYSIMD_FLOAT64_C(  -955.93), EASYSIMD_FLOAT64_C(  -801.28), EASYSIMD_FLOAT64_C(    58.35) } },
    { { EASYSIMD_FLOAT64_C(   -97.30), EASYSIMD_FLOAT64_C(   673.69), EASYSIMD_FLOAT64_C(   149.15), EASYSIMD_FLOAT64_C(  -698.57) },
      UINT8_C(  9),
      { EASYSIMD_FLOAT64_C(   444.12), EASYSIMD_FLOAT64_C(    61.54), EASYSIMD_FLOAT64_C(   565.04), EASYSIMD_FLOAT64_C(  -422.43) },
      { EASYSIMD_FLOAT64_C(   444.12), EASYSIMD_FLOAT64_C(   673.69), EASYSIMD_FLOAT64_C(   149.15), EASYSIMD_FLOAT64_C(  -422.43) } },
    { { EASYSIMD_FLOAT64_C(   503.74), EASYSIMD_FLOAT64_C(  -745.75), EASYSIMD_FLOAT64_C(  -164.57), EASYSIMD_FLOAT64_C(  -517.97) },
      UINT8_C(251),
      { EASYSIMD_FLOAT64_C(   511.61), EASYSIMD_FLOAT64_C(  -526.78), EASYSIMD_FLOAT64_C(  -278.38), EASYSIMD_FLOAT64_C(  -939.48) },
      { EASYSIMD_FLOAT64_C(   511.61), EASYSIMD_FLOAT64_C(  -526.78), EASYSIMD_FLOAT64_C(  -164.57), EASYSIMD_FLOAT64_C(  -939.48) } },
    { { EASYSIMD_FLOAT64_C(  -843.32), EASYSIMD_FLOAT64_C(   -76.55), EASYSIMD_FLOAT64_C(   382.97), EASYSIMD_FLOAT64_C(    17.16) },
      UINT8_C(137),
      { EASYSIMD_FLOAT64_C(   427.04), EASYSIMD_FLOAT64_C(   215.88), EASYSIMD_FLOAT64_C(   -96.94), EASYSIMD_FLOAT64_C(  -553.03) },
      { EASYSIMD_FLOAT64_C(   427.04), EASYSIMD_FLOAT64_C(   -76.55), EASYSIMD_FLOAT64_C(   382.97), EASYSIMD_FLOAT64_C(  -553.03) } },
    { { EASYSIMD_FLOAT64_C(   -43.34), EASYSIMD_FLOAT64_C(   -77.29), EASYSIMD_FLOAT64_C(   276.13), EASYSIMD_FLOAT64_C(  -219.33) },
      UINT8_C(215),
      { EASYSIMD_FLOAT64_C(   -50.18), EASYSIMD_FLOAT64_C(   929.82), EASYSIMD_FLOAT64_C(  -873.16), EASYSIMD_FLOAT64_C(  -986.21) },
      { EASYSIMD_FLOAT64_C(   -50.18), EASYSIMD_FLOAT64_C(   929.82), EASYSIMD_FLOAT64_C(  -873.16), EASYSIMD_FLOAT64_C(  -219.33) } },
    { { EASYSIMD_FLOAT64_C(   373.94), EASYSIMD_FLOAT64_C(   188.38), EASYSIMD_FLOAT64_C(   578.83), EASYSIMD_FLOAT64_C(   951.51) },
      UINT8_C(208),
      { EASYSIMD_FLOAT64_C(   833.07), EASYSIMD_FLOAT64_C(  -213.06), EASYSIMD_FLOAT64_C(   174.15), EASYSIMD_FLOAT64_C(  -561.72) },
      { EASYSIMD_FLOAT64_C(   373.94), EASYSIMD_FLOAT64_C(   188.38), EASYSIMD_FLOAT64_C(   578.83), EASYSIMD_FLOAT64_C(   951.51) } },
    { { EASYSIMD_FLOAT64_C(  -701.45), EASYSIMD_FLOAT64_C(   647.38), EASYSIMD_FLOAT64_C(   159.90), EASYSIMD_FLOAT64_C(  -640.93) },
      UINT8_C( 52),
      { EASYSIMD_FLOAT64_C(  -916.65), EASYSIMD_FLOAT64_C(   742.04), EASYSIMD_FLOAT64_C(  -178.78), EASYSIMD_FLOAT64_C(   928.07) },
      { EASYSIMD_FLOAT64_C(  -701.45), EASYSIMD_FLOAT64_C(   647.38), EASYSIMD_FLOAT64_C(  -178.78), EASYSIMD_FLOAT64_C(  -640.93) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256d src = easysimd_mm256_loadu_pd(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_mask_load_pd(src, k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_mask_load_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256d src = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_mask_load_pd(src, k, &a);

    easysimd_test_x86_write_f64x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_load_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint8_t k;
    EASYSIMD_ALIGN_LIKE_64(easysimd__m256d) const easysimd_float64 a[4];
    const easysimd_float64 r[4];
  } test_vec[] = {
    { UINT8_C( 56),
      { EASYSIMD_FLOAT64_C(   498.35), EASYSIMD_FLOAT64_C(   509.23), EASYSIMD_FLOAT64_C(  -793.39), EASYSIMD_FLOAT64_C(  -823.12) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -823.12) } },
    { UINT8_C(232),
      { EASYSIMD_FLOAT64_C(  -682.65), EASYSIMD_FLOAT64_C(  -327.36), EASYSIMD_FLOAT64_C(  -359.33), EASYSIMD_FLOAT64_C(  -269.17) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -269.17) } },
    { UINT8_C( 39),
      { EASYSIMD_FLOAT64_C(   502.13), EASYSIMD_FLOAT64_C(   326.36), EASYSIMD_FLOAT64_C(    21.95), EASYSIMD_FLOAT64_C(   310.52) },
      { EASYSIMD_FLOAT64_C(   502.13), EASYSIMD_FLOAT64_C(   326.36), EASYSIMD_FLOAT64_C(    21.95), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 85),
      { EASYSIMD_FLOAT64_C(  -307.32), EASYSIMD_FLOAT64_C(   193.56), EASYSIMD_FLOAT64_C(   130.23), EASYSIMD_FLOAT64_C(     7.68) },
      { EASYSIMD_FLOAT64_C(  -307.32), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(   130.23), EASYSIMD_FLOAT64_C(     0.00) } },
    { UINT8_C( 94),
      { EASYSIMD_FLOAT64_C(  -609.98), EASYSIMD_FLOAT64_C(    51.44), EASYSIMD_FLOAT64_C(   612.73), EASYSIMD_FLOAT64_C(  -970.29) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(    51.44), EASYSIMD_FLOAT64_C(   612.73), EASYSIMD_FLOAT64_C(  -970.29) } },
    { UINT8_C(139),
      { EASYSIMD_FLOAT64_C(  -923.40), EASYSIMD_FLOAT64_C(  -820.66), EASYSIMD_FLOAT64_C(   778.08), EASYSIMD_FLOAT64_C(  -121.15) },
      { EASYSIMD_FLOAT64_C(  -923.40), EASYSIMD_FLOAT64_C(  -820.66), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -121.15) } },
    { UINT8_C(204),
      { EASYSIMD_FLOAT64_C(  -643.78), EASYSIMD_FLOAT64_C(  -622.80), EASYSIMD_FLOAT64_C(  -565.72), EASYSIMD_FLOAT64_C(  -437.17) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(  -565.72), EASYSIMD_FLOAT64_C(  -437.17) } },
    { UINT8_C( 80),
      { EASYSIMD_FLOAT64_C(    32.77), EASYSIMD_FLOAT64_C(  -119.81), EASYSIMD_FLOAT64_C(   226.73), EASYSIMD_FLOAT64_C(   673.44) },
      { EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00), EASYSIMD_FLOAT64_C(     0.00) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256d r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm256_maskz_load_pd(k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm256_maskz_load_pd");
    easysimd_test_x86_assert_equal_f64x4(r, easysimd_mm256_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256d a = easysimd_test_x86_random_f64x4(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m256d r = easysimd_mm256_maskz_load_pd(k, &a);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_load_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    EASYSIMD_ALIGN_LIKE_64(easysimd__m512) const easysimd_float32 a[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   578.49), EASYSIMD_FLOAT32_C(   617.80), EASYSIMD_FLOAT32_C(  -943.28), EASYSIMD_FLOAT32_C(  -897.54),
        EASYSIMD_FLOAT32_C(   646.33), EASYSIMD_FLOAT32_C(  -832.25), EASYSIMD_FLOAT32_C(  -405.21), EASYSIMD_FLOAT32_C(   932.90),
        EASYSIMD_FLOAT32_C(  -543.24), EASYSIMD_FLOAT32_C(  -140.23), EASYSIMD_FLOAT32_C(  -849.89), EASYSIMD_FLOAT32_C(  -155.25),
        EASYSIMD_FLOAT32_C(   415.11), EASYSIMD_FLOAT32_C(   143.76), EASYSIMD_FLOAT32_C(   994.27), EASYSIMD_FLOAT32_C(   218.16) },
      { EASYSIMD_FLOAT32_C(   578.49), EASYSIMD_FLOAT32_C(   617.80), EASYSIMD_FLOAT32_C(  -943.28), EASYSIMD_FLOAT32_C(  -897.54),
        EASYSIMD_FLOAT32_C(   646.33), EASYSIMD_FLOAT32_C(  -832.25), EASYSIMD_FLOAT32_C(  -405.21), EASYSIMD_FLOAT32_C(   932.90),
        EASYSIMD_FLOAT32_C(  -543.24), EASYSIMD_FLOAT32_C(  -140.23), EASYSIMD_FLOAT32_C(  -849.89), EASYSIMD_FLOAT32_C(  -155.25),
        EASYSIMD_FLOAT32_C(   415.11), EASYSIMD_FLOAT32_C(   143.76), EASYSIMD_FLOAT32_C(   994.27), EASYSIMD_FLOAT32_C(   218.16) } },
    { { EASYSIMD_FLOAT32_C(   -52.30), EASYSIMD_FLOAT32_C(   652.12), EASYSIMD_FLOAT32_C(   377.93), EASYSIMD_FLOAT32_C(   420.34),
        EASYSIMD_FLOAT32_C(   392.51), EASYSIMD_FLOAT32_C(   983.98), EASYSIMD_FLOAT32_C(   395.81), EASYSIMD_FLOAT32_C(  -793.10),
        EASYSIMD_FLOAT32_C(  -627.87), EASYSIMD_FLOAT32_C(  -812.12), EASYSIMD_FLOAT32_C(  -637.58), EASYSIMD_FLOAT32_C(  -119.11),
        EASYSIMD_FLOAT32_C(  -504.82), EASYSIMD_FLOAT32_C(  -748.21), EASYSIMD_FLOAT32_C(   135.32), EASYSIMD_FLOAT32_C(  -926.33) },
      { EASYSIMD_FLOAT32_C(   -52.30), EASYSIMD_FLOAT32_C(   652.12), EASYSIMD_FLOAT32_C(   377.93), EASYSIMD_FLOAT32_C(   420.34),
        EASYSIMD_FLOAT32_C(   392.51), EASYSIMD_FLOAT32_C(   983.98), EASYSIMD_FLOAT32_C(   395.81), EASYSIMD_FLOAT32_C(  -793.10),
        EASYSIMD_FLOAT32_C(  -627.87), EASYSIMD_FLOAT32_C(  -812.12), EASYSIMD_FLOAT32_C(  -637.58), EASYSIMD_FLOAT32_C(  -119.11),
        EASYSIMD_FLOAT32_C(  -504.82), EASYSIMD_FLOAT32_C(  -748.21), EASYSIMD_FLOAT32_C(   135.32), EASYSIMD_FLOAT32_C(  -926.33) } },
    { { EASYSIMD_FLOAT32_C(   869.59), EASYSIMD_FLOAT32_C(   192.04), EASYSIMD_FLOAT32_C(  -823.88), EASYSIMD_FLOAT32_C(   515.92),
        EASYSIMD_FLOAT32_C(   359.79), EASYSIMD_FLOAT32_C(  -229.08), EASYSIMD_FLOAT32_C(   448.82), EASYSIMD_FLOAT32_C(   816.55),
        EASYSIMD_FLOAT32_C(   630.69), EASYSIMD_FLOAT32_C(   598.93), EASYSIMD_FLOAT32_C(  -338.70), EASYSIMD_FLOAT32_C(    45.79),
        EASYSIMD_FLOAT32_C(  -257.31), EASYSIMD_FLOAT32_C(  -344.43), EASYSIMD_FLOAT32_C(  -736.05), EASYSIMD_FLOAT32_C(   690.39) },
      { EASYSIMD_FLOAT32_C(   869.59), EASYSIMD_FLOAT32_C(   192.04), EASYSIMD_FLOAT32_C(  -823.88), EASYSIMD_FLOAT32_C(   515.92),
        EASYSIMD_FLOAT32_C(   359.79), EASYSIMD_FLOAT32_C(  -229.08), EASYSIMD_FLOAT32_C(   448.82), EASYSIMD_FLOAT32_C(   816.55),
        EASYSIMD_FLOAT32_C(   630.69), EASYSIMD_FLOAT32_C(   598.93), EASYSIMD_FLOAT32_C(  -338.70), EASYSIMD_FLOAT32_C(    45.79),
        EASYSIMD_FLOAT32_C(  -257.31), EASYSIMD_FLOAT32_C(  -344.43), EASYSIMD_FLOAT32_C(  -736.05), EASYSIMD_FLOAT32_C(   690.39) } },
    { { EASYSIMD_FLOAT32_C(  -692.31), EASYSIMD_FLOAT32_C(   641.89), EASYSIMD_FLOAT32_C(   110.73), EASYSIMD_FLOAT32_C(   700.20),
        EASYSIMD_FLOAT32_C(   625.87), EASYSIMD_FLOAT32_C(  -493.47), EASYSIMD_FLOAT32_C(   907.10), EASYSIMD_FLOAT32_C(   998.01),
        EASYSIMD_FLOAT32_C(  -305.59), EASYSIMD_FLOAT32_C(  -730.48), EASYSIMD_FLOAT32_C(  -121.10), EASYSIMD_FLOAT32_C(   189.59),
        EASYSIMD_FLOAT32_C(  -478.70), EASYSIMD_FLOAT32_C(  -985.79), EASYSIMD_FLOAT32_C(   263.25), EASYSIMD_FLOAT32_C(  -609.11) },
      { EASYSIMD_FLOAT32_C(  -692.31), EASYSIMD_FLOAT32_C(   641.89), EASYSIMD_FLOAT32_C(   110.73), EASYSIMD_FLOAT32_C(   700.20),
        EASYSIMD_FLOAT32_C(   625.87), EASYSIMD_FLOAT32_C(  -493.47), EASYSIMD_FLOAT32_C(   907.10), EASYSIMD_FLOAT32_C(   998.01),
        EASYSIMD_FLOAT32_C(  -305.59), EASYSIMD_FLOAT32_C(  -730.48), EASYSIMD_FLOAT32_C(  -121.10), EASYSIMD_FLOAT32_C(   189.59),
        EASYSIMD_FLOAT32_C(  -478.70), EASYSIMD_FLOAT32_C(  -985.79), EASYSIMD_FLOAT32_C(   263.25), EASYSIMD_FLOAT32_C(  -609.11) } },
    { { EASYSIMD_FLOAT32_C(   206.26), EASYSIMD_FLOAT32_C(   439.38), EASYSIMD_FLOAT32_C(   906.81), EASYSIMD_FLOAT32_C(  -433.95),
        EASYSIMD_FLOAT32_C(  -789.71), EASYSIMD_FLOAT32_C(   355.62), EASYSIMD_FLOAT32_C(  -617.40), EASYSIMD_FLOAT32_C(   840.98),
        EASYSIMD_FLOAT32_C(   -45.45), EASYSIMD_FLOAT32_C(    43.90), EASYSIMD_FLOAT32_C(  -113.23), EASYSIMD_FLOAT32_C(   697.24),
        EASYSIMD_FLOAT32_C(   699.47), EASYSIMD_FLOAT32_C(   150.72), EASYSIMD_FLOAT32_C(   387.62), EASYSIMD_FLOAT32_C(  -992.84) },
      { EASYSIMD_FLOAT32_C(   206.26), EASYSIMD_FLOAT32_C(   439.38), EASYSIMD_FLOAT32_C(   906.81), EASYSIMD_FLOAT32_C(  -433.95),
        EASYSIMD_FLOAT32_C(  -789.71), EASYSIMD_FLOAT32_C(   355.62), EASYSIMD_FLOAT32_C(  -617.40), EASYSIMD_FLOAT32_C(   840.98),
        EASYSIMD_FLOAT32_C(   -45.45), EASYSIMD_FLOAT32_C(    43.90), EASYSIMD_FLOAT32_C(  -113.23), EASYSIMD_FLOAT32_C(   697.24),
        EASYSIMD_FLOAT32_C(   699.47), EASYSIMD_FLOAT32_C(   150.72), EASYSIMD_FLOAT32_C(   387.62), EASYSIMD_FLOAT32_C(  -992.84) } },
    { { EASYSIMD_FLOAT32_C(  -207.39), EASYSIMD_FLOAT32_C(  -501.65), EASYSIMD_FLOAT32_C(   707.36), EASYSIMD_FLOAT32_C(  -581.51),
        EASYSIMD_FLOAT32_C(     4.88), EASYSIMD_FLOAT32_C(   614.46), EASYSIMD_FLOAT32_C(  -583.50), EASYSIMD_FLOAT32_C(   699.29),
        EASYSIMD_FLOAT32_C(   883.97), EASYSIMD_FLOAT32_C(   295.39), EASYSIMD_FLOAT32_C(  -111.12), EASYSIMD_FLOAT32_C(  -594.73),
        EASYSIMD_FLOAT32_C(   309.61), EASYSIMD_FLOAT32_C(  -847.87), EASYSIMD_FLOAT32_C(  -203.84), EASYSIMD_FLOAT32_C(  -484.14) },
      { EASYSIMD_FLOAT32_C(  -207.39), EASYSIMD_FLOAT32_C(  -501.65), EASYSIMD_FLOAT32_C(   707.36), EASYSIMD_FLOAT32_C(  -581.51),
        EASYSIMD_FLOAT32_C(     4.88), EASYSIMD_FLOAT32_C(   614.46), EASYSIMD_FLOAT32_C(  -583.50), EASYSIMD_FLOAT32_C(   699.29),
        EASYSIMD_FLOAT32_C(   883.97), EASYSIMD_FLOAT32_C(   295.39), EASYSIMD_FLOAT32_C(  -111.12), EASYSIMD_FLOAT32_C(  -594.73),
        EASYSIMD_FLOAT32_C(   309.61), EASYSIMD_FLOAT32_C(  -847.87), EASYSIMD_FLOAT32_C(  -203.84), EASYSIMD_FLOAT32_C(  -484.14) } },
    { { EASYSIMD_FLOAT32_C(   591.51), EASYSIMD_FLOAT32_C(  -297.03), EASYSIMD_FLOAT32_C(    81.91), EASYSIMD_FLOAT32_C(   801.80),
        EASYSIMD_FLOAT32_C(  -941.41), EASYSIMD_FLOAT32_C(   464.50), EASYSIMD_FLOAT32_C(   642.77), EASYSIMD_FLOAT32_C(    13.14),
        EASYSIMD_FLOAT32_C(  -491.60), EASYSIMD_FLOAT32_C(  -470.46), EASYSIMD_FLOAT32_C(  -289.62), EASYSIMD_FLOAT32_C(  -792.13),
        EASYSIMD_FLOAT32_C(   680.27), EASYSIMD_FLOAT32_C(  -902.00), EASYSIMD_FLOAT32_C(  -784.97), EASYSIMD_FLOAT32_C(  -527.12) },
      { EASYSIMD_FLOAT32_C(   591.51), EASYSIMD_FLOAT32_C(  -297.03), EASYSIMD_FLOAT32_C(    81.91), EASYSIMD_FLOAT32_C(   801.80),
        EASYSIMD_FLOAT32_C(  -941.41), EASYSIMD_FLOAT32_C(   464.50), EASYSIMD_FLOAT32_C(   642.77), EASYSIMD_FLOAT32_C(    13.14),
        EASYSIMD_FLOAT32_C(  -491.60), EASYSIMD_FLOAT32_C(  -470.46), EASYSIMD_FLOAT32_C(  -289.62), EASYSIMD_FLOAT32_C(  -792.13),
        EASYSIMD_FLOAT32_C(   680.27), EASYSIMD_FLOAT32_C(  -902.00), EASYSIMD_FLOAT32_C(  -784.97), EASYSIMD_FLOAT32_C(  -527.12) } },
    { { EASYSIMD_FLOAT32_C(  -403.65), EASYSIMD_FLOAT32_C(   922.39), EASYSIMD_FLOAT32_C(  -108.63), EASYSIMD_FLOAT32_C(   601.23),
        EASYSIMD_FLOAT32_C(   536.84), EASYSIMD_FLOAT32_C(   307.87), EASYSIMD_FLOAT32_C(   300.53), EASYSIMD_FLOAT32_C(   420.81),
        EASYSIMD_FLOAT32_C(  -396.74), EASYSIMD_FLOAT32_C(  -810.59), EASYSIMD_FLOAT32_C(   826.09), EASYSIMD_FLOAT32_C(   912.87),
        EASYSIMD_FLOAT32_C(  -658.46), EASYSIMD_FLOAT32_C(  -377.75), EASYSIMD_FLOAT32_C(  -571.27), EASYSIMD_FLOAT32_C(   933.04) },
      { EASYSIMD_FLOAT32_C(  -403.65), EASYSIMD_FLOAT32_C(   922.39), EASYSIMD_FLOAT32_C(  -108.63), EASYSIMD_FLOAT32_C(   601.23),
        EASYSIMD_FLOAT32_C(   536.84), EASYSIMD_FLOAT32_C(   307.87), EASYSIMD_FLOAT32_C(   300.53), EASYSIMD_FLOAT32_C(   420.81),
        EASYSIMD_FLOAT32_C(  -396.74), EASYSIMD_FLOAT32_C(  -810.59), EASYSIMD_FLOAT32_C(   826.09), EASYSIMD_FLOAT32_C(   912.87),
        EASYSIMD_FLOAT32_C(  -658.46), EASYSIMD_FLOAT32_C(  -377.75), EASYSIMD_FLOAT32_C(  -571.27), EASYSIMD_FLOAT32_C(   933.04) } }
 };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
      EASYSIMD_ALIGN_LIKE_64(easysimd__m512) easysimd_float32 b[16];
      easysimd_memcpy(b, test_vec[i].a, sizeof(test_vec[i].a));
      easysimd__m512 r;
      EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
        r = easysimd_mm512_load_ps(b);
      } EASYSIMD_TEST_PERF_END("easysimd_mm512_load_ps");
      easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 r = a;

    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_load_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd_float32 src[16];
    const uint16_t k;
    EASYSIMD_ALIGN_LIKE_64(easysimd__m512) const easysimd_float32 a[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { { EASYSIMD_FLOAT32_C(   260.03), EASYSIMD_FLOAT32_C(  -106.24), EASYSIMD_FLOAT32_C(   578.83), EASYSIMD_FLOAT32_C(  -408.79),
        EASYSIMD_FLOAT32_C(  -173.17), EASYSIMD_FLOAT32_C(  -181.24), EASYSIMD_FLOAT32_C(    98.90), EASYSIMD_FLOAT32_C(  -638.48),
        EASYSIMD_FLOAT32_C(    32.09), EASYSIMD_FLOAT32_C(   344.94), EASYSIMD_FLOAT32_C(  -852.21), EASYSIMD_FLOAT32_C(  -970.65),
        EASYSIMD_FLOAT32_C(  -915.06), EASYSIMD_FLOAT32_C(  -683.42), EASYSIMD_FLOAT32_C(   264.18), EASYSIMD_FLOAT32_C(   574.53) },
      UINT16_C(54473),
      { EASYSIMD_FLOAT32_C(  -454.04), EASYSIMD_FLOAT32_C(   434.66), EASYSIMD_FLOAT32_C(  -719.04), EASYSIMD_FLOAT32_C(   363.92),
        EASYSIMD_FLOAT32_C(  -812.01), EASYSIMD_FLOAT32_C(   116.81), EASYSIMD_FLOAT32_C(   910.08), EASYSIMD_FLOAT32_C(  -388.30),
        EASYSIMD_FLOAT32_C(   320.36), EASYSIMD_FLOAT32_C(    90.78), EASYSIMD_FLOAT32_C(  -678.68), EASYSIMD_FLOAT32_C(  -963.39),
        EASYSIMD_FLOAT32_C(   -65.40), EASYSIMD_FLOAT32_C(   581.35), EASYSIMD_FLOAT32_C(   -69.64), EASYSIMD_FLOAT32_C(  -486.57) },
      { EASYSIMD_FLOAT32_C(  -454.04), EASYSIMD_FLOAT32_C(  -106.24), EASYSIMD_FLOAT32_C(   578.83), EASYSIMD_FLOAT32_C(   363.92),
        EASYSIMD_FLOAT32_C(  -173.17), EASYSIMD_FLOAT32_C(  -181.24), EASYSIMD_FLOAT32_C(   910.08), EASYSIMD_FLOAT32_C(  -388.30),
        EASYSIMD_FLOAT32_C(    32.09), EASYSIMD_FLOAT32_C(   344.94), EASYSIMD_FLOAT32_C(  -678.68), EASYSIMD_FLOAT32_C(  -970.65),
        EASYSIMD_FLOAT32_C(   -65.40), EASYSIMD_FLOAT32_C(  -683.42), EASYSIMD_FLOAT32_C(   -69.64), EASYSIMD_FLOAT32_C(  -486.57) } },
    { { EASYSIMD_FLOAT32_C(  -827.44), EASYSIMD_FLOAT32_C(   757.19), EASYSIMD_FLOAT32_C(   332.19), EASYSIMD_FLOAT32_C(   271.46),
        EASYSIMD_FLOAT32_C(  -881.29), EASYSIMD_FLOAT32_C(  -635.73), EASYSIMD_FLOAT32_C(  -383.59), EASYSIMD_FLOAT32_C(  -733.50),
        EASYSIMD_FLOAT32_C(  -606.37), EASYSIMD_FLOAT32_C(  -298.65), EASYSIMD_FLOAT32_C(  -416.92), EASYSIMD_FLOAT32_C(   657.80),
        EASYSIMD_FLOAT32_C(  -724.13), EASYSIMD_FLOAT32_C(  -955.93), EASYSIMD_FLOAT32_C(   -57.68), EASYSIMD_FLOAT32_C(  -178.16) },
      UINT16_C(28624),
      { EASYSIMD_FLOAT32_C(  -814.25), EASYSIMD_FLOAT32_C(   666.72), EASYSIMD_FLOAT32_C(  -659.91), EASYSIMD_FLOAT32_C(  -904.16),
        EASYSIMD_FLOAT32_C(  -721.58), EASYSIMD_FLOAT32_C(   660.45), EASYSIMD_FLOAT32_C(   186.62), EASYSIMD_FLOAT32_C(  -400.26),
        EASYSIMD_FLOAT32_C(   697.06), EASYSIMD_FLOAT32_C(  -878.78), EASYSIMD_FLOAT32_C(  -818.91), EASYSIMD_FLOAT32_C(  -372.58),
        EASYSIMD_FLOAT32_C(  -365.35), EASYSIMD_FLOAT32_C(  -646.35), EASYSIMD_FLOAT32_C(  -615.39), EASYSIMD_FLOAT32_C(   966.83) },
      { EASYSIMD_FLOAT32_C(  -827.44), EASYSIMD_FLOAT32_C(   757.19), EASYSIMD_FLOAT32_C(   332.19), EASYSIMD_FLOAT32_C(   271.46),
        EASYSIMD_FLOAT32_C(  -721.58), EASYSIMD_FLOAT32_C(  -635.73), EASYSIMD_FLOAT32_C(   186.62), EASYSIMD_FLOAT32_C(  -400.26),
        EASYSIMD_FLOAT32_C(   697.06), EASYSIMD_FLOAT32_C(  -878.78), EASYSIMD_FLOAT32_C(  -818.91), EASYSIMD_FLOAT32_C(  -372.58),
        EASYSIMD_FLOAT32_C(  -724.13), EASYSIMD_FLOAT32_C(  -646.35), EASYSIMD_FLOAT32_C(  -615.39), EASYSIMD_FLOAT32_C(  -178.16) } },
    { { EASYSIMD_FLOAT32_C(   625.12), EASYSIMD_FLOAT32_C(  -496.68), EASYSIMD_FLOAT32_C(  -668.89), EASYSIMD_FLOAT32_C(  -758.48),
        EASYSIMD_FLOAT32_C(  -230.18), EASYSIMD_FLOAT32_C(  -275.27), EASYSIMD_FLOAT32_C(   -57.13), EASYSIMD_FLOAT32_C(   352.90),
        EASYSIMD_FLOAT32_C(  -617.46), EASYSIMD_FLOAT32_C(   218.74), EASYSIMD_FLOAT32_C(   396.97), EASYSIMD_FLOAT32_C(   324.85),
        EASYSIMD_FLOAT32_C(  -959.42), EASYSIMD_FLOAT32_C(  -124.30), EASYSIMD_FLOAT32_C(  -451.87), EASYSIMD_FLOAT32_C(  -773.66) },
      UINT16_C(60695),
      { EASYSIMD_FLOAT32_C(  -677.82), EASYSIMD_FLOAT32_C(  -179.16), EASYSIMD_FLOAT32_C(  -451.32), EASYSIMD_FLOAT32_C(   508.80),
        EASYSIMD_FLOAT32_C(   420.57), EASYSIMD_FLOAT32_C(  -754.27), EASYSIMD_FLOAT32_C(   630.01), EASYSIMD_FLOAT32_C(   601.66),
        EASYSIMD_FLOAT32_C(  -126.85), EASYSIMD_FLOAT32_C(  -735.34), EASYSIMD_FLOAT32_C(   955.32), EASYSIMD_FLOAT32_C(   257.76),
        EASYSIMD_FLOAT32_C(  -768.51), EASYSIMD_FLOAT32_C(   580.43), EASYSIMD_FLOAT32_C(   761.08), EASYSIMD_FLOAT32_C(  -437.40) },
      { EASYSIMD_FLOAT32_C(  -677.82), EASYSIMD_FLOAT32_C(  -179.16), EASYSIMD_FLOAT32_C(  -451.32), EASYSIMD_FLOAT32_C(  -758.48),
        EASYSIMD_FLOAT32_C(   420.57), EASYSIMD_FLOAT32_C(  -275.27), EASYSIMD_FLOAT32_C(   -57.13), EASYSIMD_FLOAT32_C(   352.90),
        EASYSIMD_FLOAT32_C(  -126.85), EASYSIMD_FLOAT32_C(   218.74), EASYSIMD_FLOAT32_C(   955.32), EASYSIMD_FLOAT32_C(   257.76),
        EASYSIMD_FLOAT32_C(  -959.42), EASYSIMD_FLOAT32_C(   580.43), EASYSIMD_FLOAT32_C(   761.08), EASYSIMD_FLOAT32_C(  -437.40) } },
    { { EASYSIMD_FLOAT32_C(   821.95), EASYSIMD_FLOAT32_C(  -469.10), EASYSIMD_FLOAT32_C(   287.33), EASYSIMD_FLOAT32_C(  -235.17),
        EASYSIMD_FLOAT32_C(   883.80), EASYSIMD_FLOAT32_C(   669.86), EASYSIMD_FLOAT32_C(   983.57), EASYSIMD_FLOAT32_C(   280.77),
        EASYSIMD_FLOAT32_C(    -5.28), EASYSIMD_FLOAT32_C(  -975.85), EASYSIMD_FLOAT32_C(  -843.53), EASYSIMD_FLOAT32_C(   542.85),
        EASYSIMD_FLOAT32_C(  -749.51), EASYSIMD_FLOAT32_C(  -301.11), EASYSIMD_FLOAT32_C(  -568.92), EASYSIMD_FLOAT32_C(  -427.34) },
      UINT16_C(43068),
      { EASYSIMD_FLOAT32_C(  -918.54), EASYSIMD_FLOAT32_C(   -59.70), EASYSIMD_FLOAT32_C(   225.49), EASYSIMD_FLOAT32_C(   711.47),
        EASYSIMD_FLOAT32_C(  -458.04), EASYSIMD_FLOAT32_C(  -901.36), EASYSIMD_FLOAT32_C(   976.13), EASYSIMD_FLOAT32_C(  -502.72),
        EASYSIMD_FLOAT32_C(   356.40), EASYSIMD_FLOAT32_C(  -792.38), EASYSIMD_FLOAT32_C(  -922.29), EASYSIMD_FLOAT32_C(   117.48),
        EASYSIMD_FLOAT32_C(  -229.78), EASYSIMD_FLOAT32_C(   899.67), EASYSIMD_FLOAT32_C(   648.39), EASYSIMD_FLOAT32_C(  -942.45) },
      { EASYSIMD_FLOAT32_C(   821.95), EASYSIMD_FLOAT32_C(  -469.10), EASYSIMD_FLOAT32_C(   225.49), EASYSIMD_FLOAT32_C(   711.47),
        EASYSIMD_FLOAT32_C(  -458.04), EASYSIMD_FLOAT32_C(  -901.36), EASYSIMD_FLOAT32_C(   983.57), EASYSIMD_FLOAT32_C(   280.77),
        EASYSIMD_FLOAT32_C(    -5.28), EASYSIMD_FLOAT32_C(  -975.85), EASYSIMD_FLOAT32_C(  -843.53), EASYSIMD_FLOAT32_C(   117.48),
        EASYSIMD_FLOAT32_C(  -749.51), EASYSIMD_FLOAT32_C(   899.67), EASYSIMD_FLOAT32_C(  -568.92), EASYSIMD_FLOAT32_C(  -942.45) } },
    { { EASYSIMD_FLOAT32_C(  -335.51), EASYSIMD_FLOAT32_C(   532.19), EASYSIMD_FLOAT32_C(   727.41), EASYSIMD_FLOAT32_C(  -351.94),
        EASYSIMD_FLOAT32_C(  -187.04), EASYSIMD_FLOAT32_C(  -277.87), EASYSIMD_FLOAT32_C(  -327.78), EASYSIMD_FLOAT32_C(   -30.58),
        EASYSIMD_FLOAT32_C(  -735.02), EASYSIMD_FLOAT32_C(   -77.30), EASYSIMD_FLOAT32_C(   668.31), EASYSIMD_FLOAT32_C(  -303.94),
        EASYSIMD_FLOAT32_C(   495.37), EASYSIMD_FLOAT32_C(   188.03), EASYSIMD_FLOAT32_C(   675.82), EASYSIMD_FLOAT32_C(   576.83) },
      UINT16_C(40374),
      { EASYSIMD_FLOAT32_C(   288.30), EASYSIMD_FLOAT32_C(  -329.70), EASYSIMD_FLOAT32_C(    -0.06), EASYSIMD_FLOAT32_C(   264.43),
        EASYSIMD_FLOAT32_C(   167.57), EASYSIMD_FLOAT32_C(  -643.66), EASYSIMD_FLOAT32_C(   472.05), EASYSIMD_FLOAT32_C(   245.29),
        EASYSIMD_FLOAT32_C(   473.83), EASYSIMD_FLOAT32_C(  -757.73), EASYSIMD_FLOAT32_C(   144.95), EASYSIMD_FLOAT32_C(   122.22),
        EASYSIMD_FLOAT32_C(  -700.19), EASYSIMD_FLOAT32_C(   809.44), EASYSIMD_FLOAT32_C(  -345.59), EASYSIMD_FLOAT32_C(  -972.78) },
      { EASYSIMD_FLOAT32_C(  -335.51), EASYSIMD_FLOAT32_C(  -329.70), EASYSIMD_FLOAT32_C(    -0.06), EASYSIMD_FLOAT32_C(  -351.94),
        EASYSIMD_FLOAT32_C(   167.57), EASYSIMD_FLOAT32_C(  -643.66), EASYSIMD_FLOAT32_C(  -327.78), EASYSIMD_FLOAT32_C(   245.29),
        EASYSIMD_FLOAT32_C(   473.83), EASYSIMD_FLOAT32_C(   -77.30), EASYSIMD_FLOAT32_C(   144.95), EASYSIMD_FLOAT32_C(   122.22),
        EASYSIMD_FLOAT32_C(  -700.19), EASYSIMD_FLOAT32_C(   188.03), EASYSIMD_FLOAT32_C(   675.82), EASYSIMD_FLOAT32_C(  -972.78) } },
    { { EASYSIMD_FLOAT32_C(  -542.49), EASYSIMD_FLOAT32_C(   467.37), EASYSIMD_FLOAT32_C(  -250.65), EASYSIMD_FLOAT32_C(   129.72),
        EASYSIMD_FLOAT32_C(  -563.21), EASYSIMD_FLOAT32_C(    14.33), EASYSIMD_FLOAT32_C(  -947.57), EASYSIMD_FLOAT32_C(  -894.90),
        EASYSIMD_FLOAT32_C(   710.40), EASYSIMD_FLOAT32_C(   547.79), EASYSIMD_FLOAT32_C(   293.14), EASYSIMD_FLOAT32_C(   386.21),
        EASYSIMD_FLOAT32_C(   124.62), EASYSIMD_FLOAT32_C(   421.47), EASYSIMD_FLOAT32_C(  -712.48), EASYSIMD_FLOAT32_C(  -587.08) },
      UINT16_C(19643),
      { EASYSIMD_FLOAT32_C(   677.35), EASYSIMD_FLOAT32_C(   259.34), EASYSIMD_FLOAT32_C(   643.81), EASYSIMD_FLOAT32_C(   149.40),
        EASYSIMD_FLOAT32_C(  -495.38), EASYSIMD_FLOAT32_C(   117.64), EASYSIMD_FLOAT32_C(   391.66), EASYSIMD_FLOAT32_C(   649.58),
        EASYSIMD_FLOAT32_C(  -760.14), EASYSIMD_FLOAT32_C(   691.48), EASYSIMD_FLOAT32_C(   459.02), EASYSIMD_FLOAT32_C(  -105.74),
        EASYSIMD_FLOAT32_C(   718.70), EASYSIMD_FLOAT32_C(   916.53), EASYSIMD_FLOAT32_C(  -638.37), EASYSIMD_FLOAT32_C(  -531.95) },
      { EASYSIMD_FLOAT32_C(   677.35), EASYSIMD_FLOAT32_C(   259.34), EASYSIMD_FLOAT32_C(  -250.65), EASYSIMD_FLOAT32_C(   149.40),
        EASYSIMD_FLOAT32_C(  -495.38), EASYSIMD_FLOAT32_C(   117.64), EASYSIMD_FLOAT32_C(  -947.57), EASYSIMD_FLOAT32_C(   649.58),
        EASYSIMD_FLOAT32_C(   710.40), EASYSIMD_FLOAT32_C(   547.79), EASYSIMD_FLOAT32_C(   459.02), EASYSIMD_FLOAT32_C(  -105.74),
        EASYSIMD_FLOAT32_C(   124.62), EASYSIMD_FLOAT32_C(   421.47), EASYSIMD_FLOAT32_C(  -638.37), EASYSIMD_FLOAT32_C(  -587.08) } },
    { { EASYSIMD_FLOAT32_C(    46.25), EASYSIMD_FLOAT32_C(  -201.58), EASYSIMD_FLOAT32_C(   482.39), EASYSIMD_FLOAT32_C(    98.68),
        EASYSIMD_FLOAT32_C(   -96.48), EASYSIMD_FLOAT32_C(   192.78), EASYSIMD_FLOAT32_C(  -353.53), EASYSIMD_FLOAT32_C(  -803.35),
        EASYSIMD_FLOAT32_C(  -421.00), EASYSIMD_FLOAT32_C(   771.09), EASYSIMD_FLOAT32_C(   618.12), EASYSIMD_FLOAT32_C(  -133.49),
        EASYSIMD_FLOAT32_C(  -815.99), EASYSIMD_FLOAT32_C(   709.89), EASYSIMD_FLOAT32_C(  -846.02), EASYSIMD_FLOAT32_C(   861.36) },
      UINT16_C(15304),
      { EASYSIMD_FLOAT32_C(    10.76), EASYSIMD_FLOAT32_C(   473.85), EASYSIMD_FLOAT32_C(   -84.58), EASYSIMD_FLOAT32_C(  -597.58),
        EASYSIMD_FLOAT32_C(   123.43), EASYSIMD_FLOAT32_C(   155.28), EASYSIMD_FLOAT32_C(  -906.10), EASYSIMD_FLOAT32_C(  -417.55),
        EASYSIMD_FLOAT32_C(  -950.46), EASYSIMD_FLOAT32_C(   812.60), EASYSIMD_FLOAT32_C(  -501.02), EASYSIMD_FLOAT32_C(  -588.83),
        EASYSIMD_FLOAT32_C(  -719.34), EASYSIMD_FLOAT32_C(   545.23), EASYSIMD_FLOAT32_C(   209.59), EASYSIMD_FLOAT32_C(   763.05) },
      { EASYSIMD_FLOAT32_C(    46.25), EASYSIMD_FLOAT32_C(  -201.58), EASYSIMD_FLOAT32_C(   482.39), EASYSIMD_FLOAT32_C(  -597.58),
        EASYSIMD_FLOAT32_C(   -96.48), EASYSIMD_FLOAT32_C(   192.78), EASYSIMD_FLOAT32_C(  -906.10), EASYSIMD_FLOAT32_C(  -417.55),
        EASYSIMD_FLOAT32_C(  -950.46), EASYSIMD_FLOAT32_C(   812.60), EASYSIMD_FLOAT32_C(   618.12), EASYSIMD_FLOAT32_C(  -588.83),
        EASYSIMD_FLOAT32_C(  -719.34), EASYSIMD_FLOAT32_C(   545.23), EASYSIMD_FLOAT32_C(  -846.02), EASYSIMD_FLOAT32_C(   861.36) } },
    { { EASYSIMD_FLOAT32_C(  -356.09), EASYSIMD_FLOAT32_C(  -886.89), EASYSIMD_FLOAT32_C(   -44.17), EASYSIMD_FLOAT32_C(   290.39),
        EASYSIMD_FLOAT32_C(  -690.24), EASYSIMD_FLOAT32_C(   534.83), EASYSIMD_FLOAT32_C(    61.48), EASYSIMD_FLOAT32_C(   927.88),
        EASYSIMD_FLOAT32_C(  -598.66), EASYSIMD_FLOAT32_C(   245.49), EASYSIMD_FLOAT32_C(   637.77), EASYSIMD_FLOAT32_C(  -444.68),
        EASYSIMD_FLOAT32_C(   106.85), EASYSIMD_FLOAT32_C(  -393.01), EASYSIMD_FLOAT32_C(  -646.90), EASYSIMD_FLOAT32_C(  -882.39) },
      UINT16_C(13379),
      { EASYSIMD_FLOAT32_C(  -479.97), EASYSIMD_FLOAT32_C(   204.26), EASYSIMD_FLOAT32_C(  -576.20), EASYSIMD_FLOAT32_C(  -386.07),
        EASYSIMD_FLOAT32_C(   786.71), EASYSIMD_FLOAT32_C(  -526.66), EASYSIMD_FLOAT32_C(  -573.47), EASYSIMD_FLOAT32_C(  -714.31),
        EASYSIMD_FLOAT32_C(  -115.49), EASYSIMD_FLOAT32_C(  -292.81), EASYSIMD_FLOAT32_C(   830.92), EASYSIMD_FLOAT32_C(  -905.90),
        EASYSIMD_FLOAT32_C(  -529.77), EASYSIMD_FLOAT32_C(  -525.16), EASYSIMD_FLOAT32_C(  -792.79), EASYSIMD_FLOAT32_C(   426.06) },
      { EASYSIMD_FLOAT32_C(  -479.97), EASYSIMD_FLOAT32_C(   204.26), EASYSIMD_FLOAT32_C(   -44.17), EASYSIMD_FLOAT32_C(   290.39),
        EASYSIMD_FLOAT32_C(  -690.24), EASYSIMD_FLOAT32_C(   534.83), EASYSIMD_FLOAT32_C(  -573.47), EASYSIMD_FLOAT32_C(   927.88),
        EASYSIMD_FLOAT32_C(  -598.66), EASYSIMD_FLOAT32_C(   245.49), EASYSIMD_FLOAT32_C(   830.92), EASYSIMD_FLOAT32_C(  -444.68),
        EASYSIMD_FLOAT32_C(  -529.77), EASYSIMD_FLOAT32_C(  -525.16), EASYSIMD_FLOAT32_C(  -646.90), EASYSIMD_FLOAT32_C(  -882.39) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512 src = easysimd_mm512_loadu_ps(test_vec[i].src);
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_mask_load_ps(src, k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_mask_load_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512 src = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 r = easysimd_mm512_mask_load_ps(src, k, &a);

    easysimd_test_x86_write_f32x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_load_ps (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const uint16_t k;
    EASYSIMD_ALIGN_LIKE_64(easysimd__m512) const easysimd_float32 a[16];
    const easysimd_float32 r[16];
  } test_vec[] = {
    { UINT16_C(19012),
      { EASYSIMD_FLOAT32_C(  -540.14), EASYSIMD_FLOAT32_C(   619.76), EASYSIMD_FLOAT32_C(  -276.24), EASYSIMD_FLOAT32_C(  -342.07),
        EASYSIMD_FLOAT32_C(  -366.79), EASYSIMD_FLOAT32_C(  -170.86), EASYSIMD_FLOAT32_C(   752.62), EASYSIMD_FLOAT32_C(   404.55),
        EASYSIMD_FLOAT32_C(   751.32), EASYSIMD_FLOAT32_C(  -693.33), EASYSIMD_FLOAT32_C(  -152.38), EASYSIMD_FLOAT32_C(   726.80),
        EASYSIMD_FLOAT32_C(   322.97), EASYSIMD_FLOAT32_C(  -645.15), EASYSIMD_FLOAT32_C(  -413.57), EASYSIMD_FLOAT32_C(  -553.53) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -276.24), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   752.62), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -693.33), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   726.80),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -413.57), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(17452),
      { EASYSIMD_FLOAT32_C(   714.23), EASYSIMD_FLOAT32_C(   389.19), EASYSIMD_FLOAT32_C(   638.92), EASYSIMD_FLOAT32_C(   161.90),
        EASYSIMD_FLOAT32_C(  -265.03), EASYSIMD_FLOAT32_C(   889.07), EASYSIMD_FLOAT32_C(   657.61), EASYSIMD_FLOAT32_C(  -697.03),
        EASYSIMD_FLOAT32_C(   875.44), EASYSIMD_FLOAT32_C(   624.19), EASYSIMD_FLOAT32_C(   441.70), EASYSIMD_FLOAT32_C(   433.11),
        EASYSIMD_FLOAT32_C(   -75.59), EASYSIMD_FLOAT32_C(   901.56), EASYSIMD_FLOAT32_C(    52.87), EASYSIMD_FLOAT32_C(   648.17) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   638.92), EASYSIMD_FLOAT32_C(   161.90),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   889.07), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   441.70), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    52.87), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(37108),
      { EASYSIMD_FLOAT32_C(  -522.69), EASYSIMD_FLOAT32_C(  -687.90), EASYSIMD_FLOAT32_C(    90.64), EASYSIMD_FLOAT32_C(  -771.37),
        EASYSIMD_FLOAT32_C(  -381.23), EASYSIMD_FLOAT32_C(   938.25), EASYSIMD_FLOAT32_C(   955.43), EASYSIMD_FLOAT32_C(   941.74),
        EASYSIMD_FLOAT32_C(  -706.90), EASYSIMD_FLOAT32_C(  -458.13), EASYSIMD_FLOAT32_C(  -611.80), EASYSIMD_FLOAT32_C(   471.39),
        EASYSIMD_FLOAT32_C(   167.98), EASYSIMD_FLOAT32_C(  -897.57), EASYSIMD_FLOAT32_C(  -139.42), EASYSIMD_FLOAT32_C(  -193.10) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(    90.64), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  -381.23), EASYSIMD_FLOAT32_C(   938.25), EASYSIMD_FLOAT32_C(   955.43), EASYSIMD_FLOAT32_C(   941.74),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   167.98), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -193.10) } },
    { UINT16_C(43805),
      { EASYSIMD_FLOAT32_C(  -304.02), EASYSIMD_FLOAT32_C(   -78.06), EASYSIMD_FLOAT32_C(   898.51), EASYSIMD_FLOAT32_C(  -428.58),
        EASYSIMD_FLOAT32_C(  -453.87), EASYSIMD_FLOAT32_C(   340.21), EASYSIMD_FLOAT32_C(  -995.47), EASYSIMD_FLOAT32_C(   470.54),
        EASYSIMD_FLOAT32_C(   241.77), EASYSIMD_FLOAT32_C(    57.40), EASYSIMD_FLOAT32_C(   118.70), EASYSIMD_FLOAT32_C(   801.26),
        EASYSIMD_FLOAT32_C(  -256.51), EASYSIMD_FLOAT32_C(   596.01), EASYSIMD_FLOAT32_C(  -886.64), EASYSIMD_FLOAT32_C(   834.12) },
      { EASYSIMD_FLOAT32_C(  -304.02), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   898.51), EASYSIMD_FLOAT32_C(  -428.58),
        EASYSIMD_FLOAT32_C(  -453.87), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   241.77), EASYSIMD_FLOAT32_C(    57.40), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   801.26),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   596.01), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   834.12) } },
    { UINT16_C(21329),
      { EASYSIMD_FLOAT32_C(   772.37), EASYSIMD_FLOAT32_C(   780.08), EASYSIMD_FLOAT32_C(  -326.13), EASYSIMD_FLOAT32_C(  -934.53),
        EASYSIMD_FLOAT32_C(  -678.05), EASYSIMD_FLOAT32_C(    62.08), EASYSIMD_FLOAT32_C(   536.87), EASYSIMD_FLOAT32_C(   489.93),
        EASYSIMD_FLOAT32_C(   164.51), EASYSIMD_FLOAT32_C(  -602.55), EASYSIMD_FLOAT32_C(  -703.16), EASYSIMD_FLOAT32_C(  -571.16),
        EASYSIMD_FLOAT32_C(   992.99), EASYSIMD_FLOAT32_C(    -7.19), EASYSIMD_FLOAT32_C(   350.78), EASYSIMD_FLOAT32_C(   891.51) },
      { EASYSIMD_FLOAT32_C(   772.37), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(  -678.05), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   536.87), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   164.51), EASYSIMD_FLOAT32_C(  -602.55), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   992.99), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   350.78), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(29356),
      { EASYSIMD_FLOAT32_C(   231.72), EASYSIMD_FLOAT32_C(   568.77), EASYSIMD_FLOAT32_C(   367.45), EASYSIMD_FLOAT32_C(  -526.51),
        EASYSIMD_FLOAT32_C(  -373.83), EASYSIMD_FLOAT32_C(  -513.85), EASYSIMD_FLOAT32_C(  -725.26), EASYSIMD_FLOAT32_C(   369.65),
        EASYSIMD_FLOAT32_C(  -917.83), EASYSIMD_FLOAT32_C(  -611.89), EASYSIMD_FLOAT32_C(   203.78), EASYSIMD_FLOAT32_C(   906.82),
        EASYSIMD_FLOAT32_C(   120.24), EASYSIMD_FLOAT32_C(   -23.85), EASYSIMD_FLOAT32_C(   686.90), EASYSIMD_FLOAT32_C(   794.11) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   367.45), EASYSIMD_FLOAT32_C(  -526.51),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -513.85), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   369.65),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -611.89), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   120.24), EASYSIMD_FLOAT32_C(   -23.85), EASYSIMD_FLOAT32_C(   686.90), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(24145),
      { EASYSIMD_FLOAT32_C(  -143.81), EASYSIMD_FLOAT32_C(  -421.51), EASYSIMD_FLOAT32_C(   498.78), EASYSIMD_FLOAT32_C(  -979.30),
        EASYSIMD_FLOAT32_C(   -24.06), EASYSIMD_FLOAT32_C(   795.62), EASYSIMD_FLOAT32_C(  -550.45), EASYSIMD_FLOAT32_C(   -31.07),
        EASYSIMD_FLOAT32_C(  -211.57), EASYSIMD_FLOAT32_C(   800.33), EASYSIMD_FLOAT32_C(  -139.56), EASYSIMD_FLOAT32_C(  -647.33),
        EASYSIMD_FLOAT32_C(   697.24), EASYSIMD_FLOAT32_C(  -907.84), EASYSIMD_FLOAT32_C(   921.43), EASYSIMD_FLOAT32_C(    64.69) },
      { EASYSIMD_FLOAT32_C(  -143.81), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(   -24.06), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -550.45), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   800.33), EASYSIMD_FLOAT32_C(  -139.56), EASYSIMD_FLOAT32_C(  -647.33),
        EASYSIMD_FLOAT32_C(   697.24), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   921.43), EASYSIMD_FLOAT32_C(     0.00) } },
    { UINT16_C(33316),
      { EASYSIMD_FLOAT32_C(   550.84), EASYSIMD_FLOAT32_C(  -159.61), EASYSIMD_FLOAT32_C(   917.25), EASYSIMD_FLOAT32_C(   633.01),
        EASYSIMD_FLOAT32_C(   228.50), EASYSIMD_FLOAT32_C(   121.03), EASYSIMD_FLOAT32_C(   539.83), EASYSIMD_FLOAT32_C(  -651.26),
        EASYSIMD_FLOAT32_C(  -902.82), EASYSIMD_FLOAT32_C(   226.73), EASYSIMD_FLOAT32_C(  -857.15), EASYSIMD_FLOAT32_C(   138.80),
        EASYSIMD_FLOAT32_C(   235.58), EASYSIMD_FLOAT32_C(    -0.96), EASYSIMD_FLOAT32_C(   717.29), EASYSIMD_FLOAT32_C(  -265.64) },
      { EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   917.25), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   121.03), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(   226.73), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00),
        EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(     0.00), EASYSIMD_FLOAT32_C(  -265.64) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512 r;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      r = easysimd_mm512_maskz_load_ps(k, test_vec[i].a);
    } EASYSIMD_TEST_PERF_END("easysimd_mm512_maskz_load_ps");
    easysimd_test_x86_assert_equal_f32x16(r, easysimd_mm512_loadu_ps(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512 a = easysimd_test_x86_random_f32x16(EASYSIMD_FLOAT32_C(-1000.0), EASYSIMD_FLOAT32_C(1000.0));
    easysimd__m512 r = easysimd_mm512_maskz_load_ps(k, &a);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_f32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_load_pd (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    EASYSIMD_ALIGN_LIKE_64(easysimd__m512d) const easysimd_float64 a[8];
    const easysimd_float64 r[8];
  } test_vec[] = {
    { { EASYSIMD_FLOAT64_C(   325.22), EASYSIMD_FLOAT64_C(   510.64), EASYSIMD_FLOAT64_C(   734.84), EASYSIMD_FLOAT64_C(   383.82),
        EASYSIMD_FLOAT64_C(   -24.86), EASYSIMD_FLOAT64_C(   377.61), EASYSIMD_FLOAT64_C(  -603.04), EASYSIMD_FLOAT64_C(   483.54) },
      { EASYSIMD_FLOAT64_C(   325.22), EASYSIMD_FLOAT64_C(   510.64), EASYSIMD_FLOAT64_C(   734.84), EASYSIMD_FLOAT64_C(   383.82),
        EASYSIMD_FLOAT64_C(   -24.86), EASYSIMD_FLOAT64_C(   377.61), EASYSIMD_FLOAT64_C(  -603.04), EASYSIMD_FLOAT64_C(   483.54) } },
    { { EASYSIMD_FLOAT64_C(   907.15), EASYSIMD_FLOAT64_C(   107.34), EASYSIMD_FLOAT64_C(   691.41), EASYSIMD_FLOAT64_C(   587.42),
        EASYSIMD_FLOAT64_C(   205.34), EASYSIMD_FLOAT64_C(   906.44), EASYSIMD_FLOAT64_C(  -939.70), EASYSIMD_FLOAT64_C(   801.69) },
      { EASYSIMD_FLOAT64_C(   907.15), EASYSIMD_FLOAT64_C(   107.34), EASYSIMD_FLOAT64_C(   691.41), EASYSIMD_FLOAT64_C(   587.42),
        EASYSIMD_FLOAT64_C(   205.34), EASYSIMD_FLOAT64_C(   906.44), EASYSIMD_FLOAT64_C(  -939.70), EASYSIMD_FLOAT64_C(   801.69) } },
    { { EASYSIMD_FLOAT64_C(   828.83), EASYSIMD_FLOAT64_C(   -48.33), EASYSIMD_FLOAT64_C(   402.92), EASYSIMD_FLOAT64_C(   365.67),
        EASYSIMD_FLOAT64_C(  -740.46), EASYSIMD_FLOAT64_C(  -296.55), EASYSIMD_FLOAT64_C(  -213.52), EASYSIMD_FLOAT64_C(  -137.20) },
      { EASYSIMD_FLOAT64_C(   828.83), EASYSIMD_FLOAT64_C(   -48.33), EASYSIMD_FLOAT64_C(   402.92), EASYSIMD_FLOAT64_C(   365.67),
        EASYSIMD_FLOAT64_C(  -740.46), EASYSIMD_FLOAT64_C(  -296.55), EASYSIMD_FLOAT64_C(  -213.52), EASYSIMD_FLOAT64_C(  -137.20) } },
    { { EASYSIMD_FLOAT64_C(  -107.14), EASYSIMD_FLOAT64_C(  -387.43), EASYSIMD_FLOAT64_C(  -224.34), EASYSIMD_FLOAT64_C(   234.40),
        EASYSIMD_FLOAT64_C(   234.83), EASYSIMD_FLOAT64_C(   204.39), EASYSIMD_FLOAT64_C(   167.44), EASYSIMD_FLOAT64_C(  -439.95) },
      { EASYSIMD_FLOAT64_C(  -107.14), EASYSIMD_FLOAT64_C(  -387.43), EASYSIMD_FLOAT64_C(  -224.34), EASYSIMD_FLOAT64_C(   234.40),
        EASYSIMD_FLOAT64_C(   234.83), EASYSIMD_FLOAT64_C(   204.39), EASYSIMD_FLOAT64_C(   167.44), EASYSIMD_FLOAT64_C(  -439.95) } },
    { { EASYSIMD_FLOAT64_C(  -284.97), EASYSIMD_FLOAT64_C(   -97.72), EASYSIMD_FLOAT64_C(   943.87), EASYSIMD_FLOAT64_C(   690.17),
        EASYSIMD_FLOAT64_C(  -720.11), EASYSIMD_FLOAT64_C(  -659.17), EASYSIMD_FLOAT64_C(   173.71), EASYSIMD_FLOAT64_C(  -812.95) },
      { EASYSIMD_FLOAT64_C(  -284.97), EASYSIMD_FLOAT64_C(   -97.72), EASYSIMD_FLOAT64_C(   943.87), EASYSIMD_FLOAT64_C(   690.17),
        EASYSIMD_FLOAT64_C(  -720.11), EASYSIMD_FLOAT64_C(  -659.17), EASYSIMD_FLOAT64_C(   173.71), EASYSIMD_FLOAT64_C(  -812.95) } },
    { { EASYSIMD_FLOAT64_C(   448.17), EASYSIMD_FLOAT64_C(  -134.87), EASYSIMD_FLOAT64_C(   774.47), EASYSIMD_FLOAT64_C(  -346.49),
        EASYSIMD_FLOAT64_C(  -228.43), EASYSIMD_FLOAT64_C(   834.77), EASYSIMD_FLOAT64_C(  -544.80), EASYSIMD_FLOAT64_C(  -399.60) },
      { EASYSIMD_FLOAT64_C(   448.17), EASYSIMD_FLOAT64_C(  -134.87), EASYSIMD_FLOAT64_C(   774.47), EASYSIMD_FLOAT64_C(  -346.49),
        EASYSIMD_FLOAT64_C(  -228.43), EASYSIMD_FLOAT64_C(   834.77), EASYSIMD_FLOAT64_C(  -544.80), EASYSIMD_FLOAT64_C(  -399.60) } },
    { { EASYSIMD_FLOAT64_C(  -213.56), EASYSIMD_FLOAT64_C(   858.12), EASYSIMD_FLOAT64_C(   966.07), EASYSIMD_FLOAT64_C(    45.98),
        EASYSIMD_FLOAT64_C(  -438.42), EASYSIMD_FLOAT64_C(  -247.45), EASYSIMD_FLOAT64_C(   908.78), EASYSIMD_FLOAT64_C(   454.43) },
      { EASYSIMD_FLOAT64_C(  -213.56), EASYSIMD_FLOAT64_C(   858.12), EASYSIMD_FLOAT64_C(   966.07), EASYSIMD_FLOAT64_C(    45.98),
        EASYSIMD_FLOAT64_C(  -438.42), EASYSIMD_FLOAT64_C(  -247.45), EASYSIMD_FLOAT64_C(   908.78), EASYSIMD_FLOAT64_C(   454.43) } },
    { { EASYSIMD_FLOAT64_C(   365.12), EASYSIMD_FLOAT64_C(  -315.56), EASYSIMD_FLOAT64_C(  -311.17), EASYSIMD_FLOAT64_C(  -400.05),
        EASYSIMD_FLOAT64_C(   888.84), EASYSIMD_FLOAT64_C(   856.27), EASYSIMD_FLOAT64_C(   160.00), EASYSIMD_FLOAT64_C(  -396.14) },
      { EASYSIMD_FLOAT64_C(   365.12), EASYSIMD_FLOAT64_C(  -315.56), EASYSIMD_FLOAT64_C(  -311.17), EASYSIMD_FLOAT64_C(  -400.05),
        EASYSIMD_FLOAT64_C(   888.84), EASYSIMD_FLOAT64_C(   856.27), EASYSIMD_FLOAT64_C(   160.00), EASYSIMD_FLOAT64_C(  -396.14) } }
 };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
      EASYSIMD_ALIGN_LIKE_64(easysimd__m512d) easysimd_float64 b[8];
      easysimd_memcpy(b, test_vec[i].a, sizeof(test_vec[i].a));
      easysimd__m512d r;
      EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
        r = easysimd_mm512_load_pd(b);
      } EASYSIMD_TEST_PERF_END("easysimd_mm512_load_pd");
      easysimd_test_x86_assert_equal_f64x8(r, easysimd_mm512_loadu_pd(test_vec[i].r), 1);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512d a = easysimd_test_x86_random_f64x8(EASYSIMD_FLOAT64_C(-1000.0), EASYSIMD_FLOAT64_C(1000.0));
    easysimd__m512d r = a;

    easysimd_test_x86_write_f64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_f64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_load_si512 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    EASYSIMD_ALIGN_LIKE_64(easysimd__m512i) const int32_t a[16];
    const int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(  1085279312),  INT32_C(  1689654203), -INT32_C(  1704027444),  INT32_C(  1992004399),  INT32_C(  1598136076), -INT32_C(  1107314712),  INT32_C(  1634510512),  INT32_C(  2144272078),
        -INT32_C(  1405215247), -INT32_C(    65931984), -INT32_C(  1097433201),  INT32_C(   523522579), -INT32_C(   629246223), -INT32_C(   560497363), -INT32_C(   230751453), -INT32_C(   210694911) },
      { -INT32_C(  1085279312),  INT32_C(  1689654203), -INT32_C(  1704027444),  INT32_C(  1992004399),  INT32_C(  1598136076), -INT32_C(  1107314712),  INT32_C(  1634510512),  INT32_C(  2144272078),
        -INT32_C(  1405215247), -INT32_C(    65931984), -INT32_C(  1097433201),  INT32_C(   523522579), -INT32_C(   629246223), -INT32_C(   560497363), -INT32_C(   230751453), -INT32_C(   210694911) } },
    { {  INT32_C(  1537191723),  INT32_C(   878227620),  INT32_C(  1139994160),  INT32_C(   845293376), -INT32_C(   905125475), -INT32_C(  2102877346), -INT32_C(  1468733529),  INT32_C(   547087861),
         INT32_C(   964377492),  INT32_C(   460182507),  INT32_C(    39739330),  INT32_C(   590659974),  INT32_C(    15614114), -INT32_C(  1954375964),  INT32_C(  1932785278),  INT32_C(  1888735195) },
      {  INT32_C(  1537191723),  INT32_C(   878227620),  INT32_C(  1139994160),  INT32_C(   845293376), -INT32_C(   905125475), -INT32_C(  2102877346), -INT32_C(  1468733529),  INT32_C(   547087861),
         INT32_C(   964377492),  INT32_C(   460182507),  INT32_C(    39739330),  INT32_C(   590659974),  INT32_C(    15614114), -INT32_C(  1954375964),  INT32_C(  1932785278),  INT32_C(  1888735195) } },
    { { -INT32_C(   173470198), -INT32_C(  1542383902), -INT32_C(    56201355), -INT32_C(   769664208), -INT32_C(     2945765),  INT32_C(   579491236),  INT32_C(   664125004), -INT32_C(  1751701363),
         INT32_C(   411844662), -INT32_C(   860054186),  INT32_C(  1036542733),  INT32_C(  1494279998), -INT32_C(  1722162187), -INT32_C(  2068061384),  INT32_C(   783044769), -INT32_C(  1362803848) },
      { -INT32_C(   173470198), -INT32_C(  1542383902), -INT32_C(    56201355), -INT32_C(   769664208), -INT32_C(     2945765),  INT32_C(   579491236),  INT32_C(   664125004), -INT32_C(  1751701363),
         INT32_C(   411844662), -INT32_C(   860054186),  INT32_C(  1036542733),  INT32_C(  1494279998), -INT32_C(  1722162187), -INT32_C(  2068061384),  INT32_C(   783044769), -INT32_C(  1362803848) } },
    { { -INT32_C(   624471420), -INT32_C(    56196113),  INT32_C(   607809254),  INT32_C(  1266567766),  INT32_C(  1709496109),  INT32_C(  1558880186),  INT32_C(  1737135855),  INT32_C(  1561678041),
        -INT32_C(  1858544478),  INT32_C(  1183768160), -INT32_C(  1553217459),  INT32_C(  1072621842),  INT32_C(  2057622208),  INT32_C(  1624673905), -INT32_C(    20487900),  INT32_C(  1398529201) },
      { -INT32_C(   624471420), -INT32_C(    56196113),  INT32_C(   607809254),  INT32_C(  1266567766),  INT32_C(  1709496109),  INT32_C(  1558880186),  INT32_C(  1737135855),  INT32_C(  1561678041),
        -INT32_C(  1858544478),  INT32_C(  1183768160), -INT32_C(  1553217459),  INT32_C(  1072621842),  INT32_C(  2057622208),  INT32_C(  1624673905), -INT32_C(    20487900),  INT32_C(  1398529201) } },
    { {  INT32_C(   434410425), -INT32_C(  1084263822),  INT32_C(  1281542714),  INT32_C(  1938510003), -INT32_C(  1813106654), -INT32_C(   470563650), -INT32_C(   689849819),  INT32_C(  1328102550),
         INT32_C(  1114115792), -INT32_C(  1157511040),  INT32_C(  1174889362), -INT32_C(   709258317), -INT32_C(  2123847741), -INT32_C(  1855693972), -INT32_C(  1419229931),  INT32_C(  1392218498) },
      {  INT32_C(   434410425), -INT32_C(  1084263822),  INT32_C(  1281542714),  INT32_C(  1938510003), -INT32_C(  1813106654), -INT32_C(   470563650), -INT32_C(   689849819),  INT32_C(  1328102550),
         INT32_C(  1114115792), -INT32_C(  1157511040),  INT32_C(  1174889362), -INT32_C(   709258317), -INT32_C(  2123847741), -INT32_C(  1855693972), -INT32_C(  1419229931),  INT32_C(  1392218498) } },
    { {  INT32_C(   546595743), -INT32_C(  1092905685), -INT32_C(  1425743112),  INT32_C(   947961205), -INT32_C(   776279963),  INT32_C(  1482825283), -INT32_C(   435959196), -INT32_C(    80150948),
        -INT32_C(  1927558046),  INT32_C(  1498150497),  INT32_C(  1308905433),  INT32_C(  1921483789), -INT32_C(  1354546836), -INT32_C(  1022909089), -INT32_C(   861336976),  INT32_C(  1808261385) },
      {  INT32_C(   546595743), -INT32_C(  1092905685), -INT32_C(  1425743112),  INT32_C(   947961205), -INT32_C(   776279963),  INT32_C(  1482825283), -INT32_C(   435959196), -INT32_C(    80150948),
        -INT32_C(  1927558046),  INT32_C(  1498150497),  INT32_C(  1308905433),  INT32_C(  1921483789), -INT32_C(  1354546836), -INT32_C(  1022909089), -INT32_C(   861336976),  INT32_C(  1808261385) } },
    { {  INT32_C(   251192237), -INT32_C(  1301855015), -INT32_C(  1610519661),  INT32_C(  1527941359),  INT32_C(   671765961),  INT32_C(  1810633211),  INT32_C(   624399644),  INT32_C(   613482103),
        -INT32_C(  1154250527),  INT32_C(  1617795788), -INT32_C(   184521210), -INT32_C(  1085205514),  INT32_C(  1676172136), -INT32_C(  1982933907), -INT32_C(   525466263), -INT32_C(   452641276) },
      {  INT32_C(   251192237), -INT32_C(  1301855015), -INT32_C(  1610519661),  INT32_C(  1527941359),  INT32_C(   671765961),  INT32_C(  1810633211),  INT32_C(   624399644),  INT32_C(   613482103),
        -INT32_C(  1154250527),  INT32_C(  1617795788), -INT32_C(   184521210), -INT32_C(  1085205514),  INT32_C(  1676172136), -INT32_C(  1982933907), -INT32_C(   525466263), -INT32_C(   452641276) } },
    { { -INT32_C(  1818216250), -INT32_C(   655159598),  INT32_C(  1942942588),  INT32_C(  1865555718), -INT32_C(   405661062),  INT32_C(  1483776494), -INT32_C(  1439162714),  INT32_C(   596655452),
         INT32_C(  1219899509), -INT32_C(  1155487426), -INT32_C(  1557205348), -INT32_C(  2012061683),  INT32_C(  1768940667),  INT32_C(   750903429),  INT32_C(  1540815614), -INT32_C(  1384225225) },
      { -INT32_C(  1818216250), -INT32_C(   655159598),  INT32_C(  1942942588),  INT32_C(  1865555718), -INT32_C(   405661062),  INT32_C(  1483776494), -INT32_C(  1439162714),  INT32_C(   596655452),
         INT32_C(  1219899509), -INT32_C(  1155487426), -INT32_C(  1557205348), -INT32_C(  2012061683),  INT32_C(  1768940667),  INT32_C(   750903429),  INT32_C(  1540815614), -INT32_C(  1384225225) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
      EASYSIMD_ALIGN_LIKE_64(easysimd__m512i) int32_t b[16];
      easysimd_memcpy(b, test_vec[i].a, sizeof(test_vec[i].a));
      easysimd__m512i r;
      EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
        r = easysimd_mm512_load_si512(b);
      } EASYSIMD_TEST_PERF_END("easysimd_mm512_load_si512");
      easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i r = a;

    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_load_epi32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_load_epi32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_load_epi64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_load_epi64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_load_ps)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_load_ps)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_load_pd)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_load_pd)

EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_load_epi32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_load_epi32)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_load_epi64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_load_epi64)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_load_ps)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_load_ps)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_load_pd)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_load_pd)

EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_load_ps)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_load_ps)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_load_ps)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_load_pd)
EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_load_si512)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
