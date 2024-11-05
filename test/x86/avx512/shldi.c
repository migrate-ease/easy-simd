#define EASYSIMD_TEST_X86_AVX512_INSN shldi

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/shldi.h>
#include <easysimd/easysimd-constify.h>

static int
test_easysimd_mm_shldi_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int16_t a[8];
    int16_t b[8];
    int     imm8;
    int16_t r[8];
  } test_vec[8] = {
    { { -INT16_C( 10084),  INT16_C( 16485),  INT16_C( 28424),  INT16_C( 28451), -INT16_C(  6803),  INT16_C( 14024), -INT16_C( 18272),  INT16_C( 22104) },
      {  INT16_C( 26311),  INT16_C( 29589), -INT16_C(  1217),  INT16_C( 27683),  INT16_C( 14721), -INT16_C( 14984), -INT16_C(  3881),  INT16_C( 29833) },
       INT32_C(           8),
      { -INT16_C( 25498),  INT16_C( 25971),  INT16_C(  2299),  INT16_C(  9068),  INT16_C( 27961), -INT16_C( 14139), -INT16_C( 24336),  INT16_C( 22644) } },
    { { -INT16_C( 10402), -INT16_C( 13504),  INT16_C(  2236),  INT16_C( 23553),  INT16_C( 23232), -INT16_C( 30797),  INT16_C( 18624),  INT16_C(   251) },
      {  INT16_C(  7747), -INT16_C( 14996), -INT16_C(  7081),  INT16_C( 11914),  INT16_C(  5076), -INT16_C( 25438),  INT16_C( 22274),  INT16_C( 24685) },
       INT32_C(          46),
      { -INT16_C( 30832),  INT16_C( 12635),  INT16_C( 14613),  INT16_C( 19362),  INT16_C(  1269), -INT16_C(  6360),  INT16_C(  5568), -INT16_C( 10213) } },
    { {  INT16_C( 11701),  INT16_C( 30279), -INT16_C(  1401),  INT16_C( 18429), -INT16_C(  1982), -INT16_C( 31161), -INT16_C( 19690),  INT16_C( 27979) },
      { -INT16_C( 10857),  INT16_C( 27548),  INT16_C( 16104), -INT16_C(  5624),  INT16_C( 30101), -INT16_C( 15286),  INT16_C( 30243), -INT16_C( 10065) },
       INT32_C(          35),
      {  INT16_C( 28078), -INT16_C( 19909), -INT16_C( 11207),  INT16_C( 16367), -INT16_C( 15853),  INT16_C( 12862), -INT16_C( 26445),  INT16_C( 27230) } },
    { {  INT16_C( 19697),  INT16_C( 13169), -INT16_C( 18108),  INT16_C( 23481),  INT16_C(  1132),  INT16_C(  1224),  INT16_C( 25817), -INT16_C( 15761) },
      {  INT16_C( 30627),  INT16_C( 14508), -INT16_C(  2067),  INT16_C(  4348), -INT16_C( 21651),  INT16_C(  4328),  INT16_C( 14242), -INT16_C( 27846) },
       INT32_C(           3),
      {  INT16_C( 26507), -INT16_C( 25719), -INT16_C( 13785), -INT16_C(  8760),  INT16_C(  9061),  INT16_C(  9792),  INT16_C(  9929),  INT16_C(  4988) } },
    { { -INT16_C( 32668), -INT16_C( 11998), -INT16_C(  5244),  INT16_C( 24277),  INT16_C( 17487), -INT16_C(  3552), -INT16_C( 13124), -INT16_C( 22229) },
      {  INT16_C( 10179),  INT16_C( 12473), -INT16_C( 24109),  INT16_C( 30016),  INT16_C( 31448),  INT16_C( 23304), -INT16_C( 12762), -INT16_C( 30173) },
       INT32_C(          14),
      {  INT16_C(  2544), -INT16_C( 29650),  INT16_C( 10356),  INT16_C( 23888), -INT16_C(  8522),  INT16_C(  5826),  INT16_C( 13193), -INT16_C(  7544) } },
    { {  INT16_C( 12336), -INT16_C( 32719),  INT16_C( 20853),  INT16_C( 12658), -INT16_C( 25315), -INT16_C(  7718), -INT16_C( 27707), -INT16_C( 26607) },
      {  INT16_C( 21044),  INT16_C(  3341),  INT16_C(  5580), -INT16_C(  3480), -INT16_C( 29725),  INT16_C( 12925), -INT16_C( 10031),  INT16_C(   261) },
       INT32_C(           9),
      {  INT16_C( 24740),  INT16_C( 25114), -INT16_C(  5589), -INT16_C(  6684),  INT16_C( 15127), -INT16_C( 19356), -INT16_C( 29775),  INT16_C(  8706) } },
    { { -INT16_C(  2937), -INT16_C( 23377), -INT16_C( 30319),  INT16_C( 22149), -INT16_C( 26852),  INT16_C( 20718), -INT16_C(  1047), -INT16_C( 19107) },
      { -INT16_C( 14832), -INT16_C(  2904),  INT16_C(  9553),  INT16_C(  8742),  INT16_C( 11261),  INT16_C(  1572), -INT16_C( 23199), -INT16_C(  6012) },
       INT32_C(          25),
      {  INT16_C(  3980),  INT16_C( 24553),  INT16_C(  8778),  INT16_C(  2628),  INT16_C( 14423), -INT16_C(  9204), -INT16_C( 11446), -INT16_C( 17455) } },
    { {  INT16_C(  4796), -INT16_C( 10111),  INT16_C( 28841), -INT16_C( 28119), -INT16_C( 31125),  INT16_C( 31815), -INT16_C(  4276), -INT16_C( 24976) },
      { -INT16_C( 27116),  INT16_C(  4800), -INT16_C(  6975),  INT16_C(  8728), -INT16_C( 25206),  INT16_C(  8970), -INT16_C( 26928), -INT16_C( 29362) },
       INT32_C(          40),
      { -INT16_C( 17258), -INT16_C( 32494), -INT16_C( 22044),  INT16_C( 10530),  INT16_C( 27549),  INT16_C( 18211),  INT16_C( 19606),  INT16_C( 28813) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    int       imm8 = test_vec[i].imm8;
    easysimd__m128i r = easysimd_mm_loadu_si128(test_vec[i].r);
    easysimd__m128i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      EASYSIMD_CONSTIFY_64_(easysimd_mm_shldi_epi16, ret, easysimd_mm_setzero_si128(), imm8, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_shldi_epi16");
    easysimd_assert_m128i_i16(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    int       imm8 = easysimd_test_codegen_random_i32() & 63;
    easysimd__m128i r;
    EASYSIMD_CONSTIFY_64_(easysimd_mm_shldi_epi16, r, easysimd_mm_setzero_si128(), imm8, a, b);

    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_shldi_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t a[4];
    int32_t b[4];
    int     imm8;
    int32_t r[4];
  } test_vec[8] = {
    { { -INT32_C(  1993194296), -INT32_C(   305105754), -INT32_C(  1833203025),  INT32_C(  1819488077) },
      {  INT32_C(   268255706),  INT32_C(  1129535976),  INT32_C(   255536701),  INT32_C(  1200432255) },
       INT32_C(          41),
      {  INT32_C(  1686736927), -INT32_C(  1595323258),  INT32_C(  1997889054), -INT32_C(   430007665) } },
    { { -INT32_C(   469982925),  INT32_C(  2020980523), -INT32_C(  1041962521),  INT32_C(   315679018) },
      {  INT32_C(  2018911290), -INT32_C(   779644590),  INT32_C(   521671862),  INT32_C(   120515027) },
       INT32_C(          10),
      { -INT32_C(   226177567), -INT32_C(   690180282), -INT32_C(  1817731972),  INT32_C(  1132767260) } },
    { { -INT32_C(   919773214),  INT32_C(  1921651016),  INT32_C(   763714291), -INT32_C(   777659522) },
      {  INT32_C(   581053548),  INT32_C(   339852096),  INT32_C(   773549988),  INT32_C(  2095252890) },
       INT32_C(          36),
      { -INT32_C(  1831469534),  INT32_C(   681645185), -INT32_C(   665473230),  INT32_C(   442349543) } },
    { {  INT32_C(   337628961), -INT32_C(  1472027607), -INT32_C(   327555201),  INT32_C(  1410210580) },
      {  INT32_C(  2053656790),  INT32_C(  1504215999), -INT32_C(   304772216), -INT32_C(  1113974117) },
       INT32_C(          41),
      {  INT32_C(  1067336436), -INT32_C(  2058857805), -INT32_C(   204537893),  INT32_C(   473311611) } },
    { { -INT32_C(   574942370),  INT32_C(   264844539),  INT32_C(   644077647), -INT32_C(   408892376) },
      { -INT32_C(   666810033),  INT32_C(  1891964628),  INT32_C(   439181105),  INT32_C(  1982725656) },
       INT32_C(          18),
      {  INT32_C(  1299931397), -INT32_C(   739392748),  INT32_C(  1631348917),  INT32_C(   815913143) } },
    { {  INT32_C(  1813781788),  INT32_C(   496140277), -INT32_C(  1694158261),  INT32_C(  1349731963) },
      { -INT32_C(  1916782500), -INT32_C(  1347883625), -INT32_C(    47786517), -INT32_C(   636847426) },
       INT32_C(          23),
      { -INT32_C(  1907957732), -INT32_C(    86518666),  INT32_C(   637440874),  INT32_C(  1038943549) } },
    { { -INT32_C(   240461658), -INT32_C(  2037600501),  INT32_C(  1389821941), -INT32_C(   824207817) },
      {  INT32_C(  1870563203),  INT32_C(   460104797), -INT32_C(  1242204386),  INT32_C(  1128414365) },
       INT32_C(          20),
      { -INT32_C(  1972963352), -INT32_C(   256788790), -INT32_C(    10789033),  INT32_C(  1668559907) } },
    { { -INT32_C(  1834565220), -INT32_C(   119243583), -INT32_C(  1765293293), -INT32_C(  1476049589) },
      {  INT32_C(   130249449), -INT32_C(  2034452247),  INT32_C(   164298740), -INT32_C(  2010578964) },
       INT32_C(           0),
      { -INT32_C(  1834565220), -INT32_C(   119243583), -INT32_C(  1765293293), -INT32_C(  1476049589) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    int       imm8 = test_vec[i].imm8;
    easysimd__m128i r = easysimd_mm_loadu_si128(test_vec[i].r);
    easysimd__m128i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      EASYSIMD_CONSTIFY_64_(easysimd_mm_shldi_epi32, ret, easysimd_mm_setzero_si128(), imm8, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_shldi_epi32");
    easysimd_assert_m128i_i32(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    int       imm8 = easysimd_test_codegen_random_i32() & 63;
    easysimd__m128i r;
    EASYSIMD_CONSTIFY_64_(easysimd_mm_shldi_epi32, r, easysimd_mm_setzero_si128(), imm8, a, b);

    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_shldi_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t a[2];
    int64_t b[2];
    int     imm8;
    int64_t r[2];
  } test_vec[8] = {
    { { -INT64_C( 2265932828931669920), -INT64_C( 1445400025876297402) },
      { -INT64_C( 2108721454315367196), -INT64_C( 6047392788895911386) },
       INT32_C(          24),
      {  INT64_C( 1798802948812356689),  INT64_C( 6170861166419252056) } },
    { {  INT64_C( 1038464241542115455), -INT64_C( 1176772191736735419) },
      {  INT64_C( 7296793579523041172), -INT64_C( 8499500809206999357) },
       INT32_C(          37),
      { -INT64_C( 6771707480592522441),  INT64_C( 4551776590160634235) } },
    { { -INT64_C(    6855374507558536),  INT64_C( 1312731093705659112) },
      { -INT64_C( 2537063905311098324),  INT64_C( 7548516692463703358) },
       INT32_C(          48),
      { -INT64_C( 5082069417033863305), -INT64_C( 7860917968351255705) } },
    { { -INT64_C( 7532146548650472429), -INT64_C(  338859406115885007) },
      { -INT64_C( 2662682997209688406),  INT64_C( 5685910233394958070) },
       INT32_C(          53),
      { -INT64_C( 9044528189942187262),  INT64_C(  444129086838458489) } },
    { { -INT64_C( 7104052988889808166), -INT64_C( 5257547067955825290) },
      {  INT64_C( 2237117917438351023),  INT64_C( 4019946855020294975) },
       INT32_C(          58),
      {  INT64_C( 7528944747404479578), -INT64_C( 2819492091907425332) } },
    { {  INT64_C( 3269651130544204745),  INT64_C( 7199135392818555283) },
      {  INT64_C( 4918759719855977294),  INT64_C( 5264916322444839849) },
       INT32_C(          41),
      {  INT64_C( 1537858912829820699), -INT64_C( 3975791636969226170) } },
    { { -INT64_C( 3273312832552703555),  INT64_C( 6138046127443114579) },
      { -INT64_C(  254993804705566295), -INT64_C( 2448604555764136239) },
       INT32_C(          56),
      { -INT64_C( 4756797276052874895),  INT64_C( 6043273037639992966) } },
    { { -INT64_C( 6211851215695986473), -INT64_C( 1745541322074293814) },
      { -INT64_C( 1966056792733303637), -INT64_C( 8161665227943658504) },
       INT32_C(          48),
      { -INT64_C( 4550917423103771421), -INT64_C(  447388275121344318) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    int       imm8 = test_vec[i].imm8;
    easysimd__m128i r = easysimd_mm_loadu_si128(test_vec[i].r);
    easysimd__m128i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      EASYSIMD_CONSTIFY_64_(easysimd_mm_shldi_epi64, ret, easysimd_mm_setzero_si128(), imm8, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_shldi_epi64");
    easysimd_assert_m128i_i64(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    int       imm8 = easysimd_test_codegen_random_i32() & 63;
    easysimd__m128i r;
    EASYSIMD_CONSTIFY_64_(easysimd_mm_shldi_epi64, r, easysimd_mm_setzero_si128(), imm8, a, b);

    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_shldi_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int16_t a[16];
    int16_t b[16];
    int     imm8;
    int16_t r[16];
  } test_vec[8] = {
    { { -INT16_C(  7748),  INT16_C( 12120), -INT16_C( 20788),  INT16_C(   642),  INT16_C(  2411), -INT16_C(  7635),  INT16_C( 21280), -INT16_C( 13836),
        -INT16_C(  5608),  INT16_C( 20728), -INT16_C( 17746),  INT16_C( 14138),  INT16_C( 26246), -INT16_C( 28980), -INT16_C( 22380),  INT16_C( 20734) },
      {  INT16_C( 22153),  INT16_C( 21887),  INT16_C(   516),  INT16_C( 28503), -INT16_C( 31733),  INT16_C( 11090),  INT16_C( 18135), -INT16_C(  3852),
        -INT16_C(  5072), -INT16_C(  8640),  INT16_C( 31655),  INT16_C( 11541), -INT16_C(  7711),  INT16_C( 30139), -INT16_C( 18039),  INT16_C(  4805) },
       INT32_C(          15),
      {  INT16_C( 11076),  INT16_C( 10943),  INT16_C(   258),  INT16_C( 14251), -INT16_C( 15867), -INT16_C( 27223),  INT16_C(  9067),  INT16_C( 30842),
         INT16_C( 30232),  INT16_C( 28448),  INT16_C( 15827),  INT16_C(  5770),  INT16_C( 28912),  INT16_C( 15069),  INT16_C( 23748),  INT16_C(  2402) } },
    { { -INT16_C( 16569),  INT16_C( 21123), -INT16_C( 10941),  INT16_C(  7038),  INT16_C( 29211),  INT16_C( 19211),  INT16_C( 19295),  INT16_C(  1577),
         INT16_C( 16070), -INT16_C( 22733), -INT16_C(  4577), -INT16_C( 22500), -INT16_C(  7512), -INT16_C( 18502),  INT16_C(  8743),  INT16_C( 28363) },
      {  INT16_C( 20449),  INT16_C(  9408),  INT16_C( 15908),  INT16_C( 16447),  INT16_C( 19121),  INT16_C(  4235), -INT16_C( 19050),  INT16_C( 23574),
         INT16_C( 18931),  INT16_C(  4868),  INT16_C(  8247), -INT16_C(  8261),  INT16_C( 30210),  INT16_C( 10647),  INT16_C( 25240),  INT16_C( 31127) },
       INT32_C(          49),
      {  INT16_C( 32398), -INT16_C( 23290), -INT16_C( 21882),  INT16_C( 14076), -INT16_C(  7114), -INT16_C( 27114), -INT16_C( 26945),  INT16_C(  3154),
         INT16_C( 32140),  INT16_C( 20070), -INT16_C(  9154),  INT16_C( 20537), -INT16_C( 15024),  INT16_C( 28532),  INT16_C( 17486), -INT16_C(  8810) } },
    { { -INT16_C(  8810),  INT16_C( 18198), -INT16_C( 24281), -INT16_C( 17065),  INT16_C( 27990),  INT16_C( 18970),  INT16_C(  7862), -INT16_C(  4515),
         INT16_C(  6206),  INT16_C( 16845),  INT16_C( 25742),  INT16_C(  9834),  INT16_C(   711),  INT16_C( 30879),  INT16_C( 15706), -INT16_C(  4018) },
      {  INT16_C( 25626),  INT16_C( 16696), -INT16_C( 28922),  INT16_C( 23807),  INT16_C(  6653), -INT16_C( 19546),  INT16_C(   823),  INT16_C( 30113),
         INT16_C( 28444), -INT16_C( 21834),  INT16_C(  8659), -INT16_C( 25903),  INT16_C( 28707),  INT16_C( 32019),  INT16_C( 25005), -INT16_C( 14483) },
       INT32_C(           6),
      {  INT16_C( 26009), -INT16_C( 14960),  INT16_C( 18915),  INT16_C( 21975),  INT16_C( 21894), -INT16_C( 31060), -INT16_C( 21120), -INT16_C( 26787),
         INT16_C(  3995),  INT16_C( 29546),  INT16_C(  9096), -INT16_C( 25946), -INT16_C( 20004),  INT16_C( 10207),  INT16_C( 22168),  INT16_C(  5041) } },
    { {  INT16_C(  2101),  INT16_C( 12840), -INT16_C( 12511),  INT16_C( 22757), -INT16_C( 30766), -INT16_C(  4403), -INT16_C( 31498), -INT16_C( 13927),
         INT16_C( 27301), -INT16_C( 14236),  INT16_C( 30682), -INT16_C( 30651), -INT16_C( 19752), -INT16_C( 25009),  INT16_C( 22616), -INT16_C( 29334) },
      { -INT16_C( 27808), -INT16_C( 32321), -INT16_C( 23454),  INT16_C( 13529), -INT16_C( 22741),  INT16_C(  8483), -INT16_C( 17365), -INT16_C( 12053),
         INT16_C( 20262),  INT16_C(   152), -INT16_C(  8762), -INT16_C( 24952), -INT16_C( 10097), -INT16_C(  6339), -INT16_C( 22736), -INT16_C( 28300) },
       INT32_C(          58),
      { -INT16_C( 10675), -INT16_C( 24058), -INT16_C( 31087), -INT16_C( 27437),  INT16_C( 19100),  INT16_C( 13444), -INT16_C(  9488),  INT16_C( 26435),
        -INT16_C( 27332), -INT16_C( 28670),  INT16_C( 27511),  INT16_C(  5754),  INT16_C( 25442),  INT16_C( 16284),  INT16_C( 25244), -INT16_C( 21947) } },
    { { -INT16_C(  4904),  INT16_C(   977), -INT16_C(  2925), -INT16_C( 16859),  INT16_C(  4272), -INT16_C( 10610),  INT16_C(  9823),  INT16_C(  9686),
         INT16_C( 24323), -INT16_C( 27965),  INT16_C(    55),  INT16_C( 26490), -INT16_C(  4440), -INT16_C(  7432),  INT16_C(  2850), -INT16_C(  1409) },
      {  INT16_C( 20727), -INT16_C( 29955),  INT16_C(  8772), -INT16_C(  3000), -INT16_C( 10702), -INT16_C( 28214), -INT16_C( 24324), -INT16_C(    74),
         INT16_C( 31487),  INT16_C( 13969),  INT16_C(  2938),  INT16_C(  8862), -INT16_C( 26886),  INT16_C(  7173), -INT16_C( 31583), -INT16_C( 26602) },
       INT32_C(          20),
      { -INT16_C( 12923),  INT16_C( 15640),  INT16_C( 18738), -INT16_C(  7585),  INT16_C(  2829),  INT16_C( 26857),  INT16_C( 26106),  INT16_C( 23919),
        -INT16_C(  4041),  INT16_C( 11315),  INT16_C(   880),  INT16_C( 30626), -INT16_C(  5495),  INT16_C( 12161), -INT16_C( 19928), -INT16_C( 22535) } },
    { {  INT16_C( 27190),  INT16_C( 26636), -INT16_C( 10688),  INT16_C( 15610), -INT16_C( 20362),  INT16_C( 30267), -INT16_C( 13014), -INT16_C( 23124),
         INT16_C( 19160), -INT16_C( 11577), -INT16_C( 13087), -INT16_C( 32018),  INT16_C(  1104),  INT16_C(  9243),  INT16_C( 15640),  INT16_C( 20028) },
      {  INT16_C( 18600), -INT16_C(  5962), -INT16_C( 20450), -INT16_C( 27355),  INT16_C( 24673), -INT16_C( 29941), -INT16_C( 18643),  INT16_C(  1584),
        -INT16_C(  2046), -INT16_C(  7208), -INT16_C( 14396),  INT16_C(  5477), -INT16_C( 32565), -INT16_C(  7367),  INT16_C( 30398),  INT16_C( 26161) },
       INT32_C(          62),
      { -INT16_C( 28118),  INT16_C( 14893),  INT16_C( 11271), -INT16_C( 23223), -INT16_C( 26600), -INT16_C(  7486), -INT16_C( 21045),  INT16_C(   396),
         INT16_C( 15872), -INT16_C(  1802),  INT16_C( 29169), -INT16_C( 31399),  INT16_C(  8242), -INT16_C(  1842),  INT16_C(  7599),  INT16_C(  6540) } },
    { {  INT16_C( 29592), -INT16_C(  1678),  INT16_C( 32212),  INT16_C(   389), -INT16_C( 19148),  INT16_C( 13831), -INT16_C(  8019),  INT16_C( 29209),
         INT16_C( 32679),  INT16_C( 29319), -INT16_C( 16129), -INT16_C( 17066), -INT16_C( 30922), -INT16_C(  2781),  INT16_C( 29295),  INT16_C(  2258) },
      {  INT16_C( 17637), -INT16_C( 18175), -INT16_C( 31039), -INT16_C(  2629), -INT16_C( 15812), -INT16_C(  5844),  INT16_C( 17826),  INT16_C( 18779),
        -INT16_C(  7484), -INT16_C( 15172),  INT16_C(  4771), -INT16_C(  9855), -INT16_C( 23143),  INT16_C(  2510), -INT16_C( 24553), -INT16_C(  1007) },
       INT32_C(          36),
      {  INT16_C( 14724), -INT16_C( 26837), -INT16_C(  8888),  INT16_C(  6239),  INT16_C( 21324),  INT16_C( 24702),  INT16_C(  2772),  INT16_C(  8596),
        -INT16_C(  1410),  INT16_C( 10364),  INT16_C(  4081), -INT16_C( 10899),  INT16_C( 29546),  INT16_C( 21040),  INT16_C(  9978), -INT16_C( 29393) } },
    { {  INT16_C( 29081), -INT16_C( 10853), -INT16_C( 14541), -INT16_C( 10562),  INT16_C(  6668), -INT16_C( 12001), -INT16_C(  9220), -INT16_C( 24683),
         INT16_C(  5869), -INT16_C( 30855),  INT16_C( 18363), -INT16_C( 11632), -INT16_C( 24088), -INT16_C( 13105), -INT16_C( 31309),  INT16_C( 19570) },
      {  INT16_C(  3574),  INT16_C( 10529), -INT16_C(  7980), -INT16_C(  7937),  INT16_C(  8186), -INT16_C(  2383),  INT16_C( 18170), -INT16_C(  5994),
         INT16_C(  3933),  INT16_C(  6255), -INT16_C(   170),  INT16_C( 16107), -INT16_C( 17760),  INT16_C( 21259),  INT16_C( 32063),  INT16_C( 13728) },
       INT32_C(          10),
      {  INT16_C( 25655),  INT16_C( 27812), -INT16_C( 12413), -INT16_C(  1149),  INT16_C( 12415),  INT16_C( 32730), -INT16_C(  3813),  INT16_C( 22434),
        -INT16_C( 19395), -INT16_C(  7071), -INT16_C(  4099),  INT16_C( 16635), -INT16_C( 23830),  INT16_C( 15692), -INT16_C( 12812), -INT16_C( 14122) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    int       imm8 = test_vec[i].imm8;
    easysimd__m256i r = easysimd_mm256_loadu_si256(test_vec[i].r);
    easysimd__m256i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      EASYSIMD_CONSTIFY_64_(easysimd_mm256_shldi_epi16, ret, easysimd_mm256_setzero_si256(), imm8, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_shldi_epi16");
    easysimd_assert_m256i_i16(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i b = easysimd_test_x86_random_i16x16();
    int       imm8 = easysimd_test_codegen_random_i32() & 63;
    easysimd__m256i r;
    EASYSIMD_CONSTIFY_64_(easysimd_mm256_shldi_epi16, r, easysimd_mm256_setzero_si256(), imm8, a, b);

    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_shldi_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t a[8];
    int32_t b[8];
    int     imm8;
    int32_t r[8];
  } test_vec[8] = {
    { { -INT32_C(   620049436),  INT32_C(  1699978755),  INT32_C(   458255146),  INT32_C(  1839444295),  INT32_C(  1862829244), -INT32_C(  1826542273),  INT32_C(  1290419077),  INT32_C(  1147381856) },
      {  INT32_C(  1998548596),  INT32_C(   937194252),  INT32_C(   642919902), -INT32_C(  1701579298), -INT32_C(  1190552711),  INT32_C(  1296837320), -INT32_C(   979749275),  INT32_C(  1393163743) },
       INT32_C(          43),
      {  INT32_C(  1449075640), -INT32_C(  1661986370), -INT32_C(  2091298510),  INT32_C(   495598804),  INT32_C(  1143334344),  INT32_C(   157940330),  INT32_C(  1373384236),  INT32_C(   490930840) } },
    { {  INT32_C(  2058332060), -INT32_C(  1298136620),  INT32_C(  1867265270), -INT32_C(  1742187057), -INT32_C(   437947009), -INT32_C(  1968537685), -INT32_C(   388123523),  INT32_C(  2036377821) },
      {  INT32_C(   603131727),  INT32_C(   114660368), -INT32_C(  1753865784), -INT32_C(   181428618), -INT32_C(  1076226797),  INT32_C(   290030996),  INT32_C(   385492793),  INT32_C(   512711375) },
       INT32_C(          42),
      { -INT32_C(  1096912753), -INT32_C(  2147004389),  INT32_C(   819190365), -INT32_C(  1588117548), -INT32_C(  1781137665), -INT32_C(  1442927547),  INT32_C(  1993471067), -INT32_C(  2103217030) } },
    { { -INT32_C(   561965546), -INT32_C(  1384712393), -INT32_C(  1448958571),  INT32_C(  1332247994),  INT32_C(   996192514), -INT32_C(  1487840552),  INT32_C(   533061813),  INT32_C(  2040071778) },
      {  INT32_C(  1398217244), -INT32_C(  1493119727),  INT32_C(   760259443),  INT32_C(   578598944),  INT32_C(  1113447785), -INT32_C(   320229833), -INT32_C(   267669618), -INT32_C(   764828235) },
       INT32_C(          63),
      {  INT32_C(   699108622), -INT32_C(   746559864), -INT32_C(  1767353927),  INT32_C(   289299472),  INT32_C(   556723892),  INT32_C(  1987368731), -INT32_C(   133834809),  INT32_C(  1765069530) } },
    { {  INT32_C(    24651406), -INT32_C(   382744375), -INT32_C(   385111168), -INT32_C(  1070897016), -INT32_C(  1532226282),  INT32_C(  2039789764),  INT32_C(   474742365),  INT32_C(  1307406783) },
      {  INT32_C(  1615816087), -INT32_C(  1387692499), -INT32_C(  1298705367), -INT32_C(   764230980), -INT32_C(  1686757673),  INT32_C(   857017046), -INT32_C(   951033848),  INT32_C(  1746222545) },
       INT32_C(          35),
      {  INT32_C(   197211251),  INT32_C(  1233012301),  INT32_C(  1214077957),  INT32_C(    22758470),  INT32_C(   627091636), -INT32_C(   861551071), -INT32_C(   497028370),  INT32_C(  1869319675) } },
    { {  INT32_C(   192811490),  INT32_C(   566039909), -INT32_C(  1343017001),  INT32_C(   608856398),  INT32_C(  2086166388), -INT32_C(  1857771328), -INT32_C(  1996858907), -INT32_C(  1621507395) },
      {  INT32_C(   967497684), -INT32_C(  1017419540), -INT32_C(   445493609),  INT32_C(   738901432), -INT32_C(   592944612), -INT32_C(   277943286),  INT32_C(    41445445), -INT32_C(    22949590) },
       INT32_C(          40),
      {  INT32_C(  2115101241), -INT32_C(  1122671165), -INT32_C(   214968347),  INT32_C(  1248415276),  INT32_C(  1482650844),  INT32_C(  1151910127), -INT32_C(    94771966),  INT32_C(  1505934846) } },
    { {  INT32_C(  1264030644), -INT32_C(  1724790047), -INT32_C(  1547355257), -INT32_C(  1484755299), -INT32_C(  1600721318), -INT32_C(  2136863146), -INT32_C(  2021702433),  INT32_C(  1142667152) },
      {  INT32_C(   730821450), -INT32_C(   993673155), -INT32_C(  1738044677),  INT32_C(  1396697080),  INT32_C(   737400533), -INT32_C(   995322396),  INT32_C(  1766534105),  INT32_C(   749561826) },
       INT32_C(          26),
      { -INT32_C(   793887283), -INT32_C(  2028792064),  INT32_C(   509713963),  INT32_C(  1967980447),  INT32_C(  1756352347),  INT32_C(  1527951959),  INT32_C(  2107976879),  INT32_C(  1085453727) } },
    { { -INT32_C(   103080707), -INT32_C(  1617870169), -INT32_C(      863958), -INT32_C(  1960057433),  INT32_C(  1414518651), -INT32_C(   457270526), -INT32_C(   602903806), -INT32_C(  1493997656) },
      {  INT32_C(   731893380),  INT32_C(   986329104), -INT32_C(  1472545535),  INT32_C(   489973154),  INT32_C(  1047692092),  INT32_C(   539111454),  INT32_C(  1157378715),  INT32_C(   501936025) },
       INT32_C(          61),
      { -INT32_C(  1519126064), -INT32_C(   413579774),  INT32_C(  1426544544), -INT32_C(   475624268),  INT32_C(  1741574247),  INT32_C(  1141130755),  INT32_C(  1218414163),  INT32_C(    62742003) } },
    { { -INT32_C(  1157098567),  INT32_C(  1919107792), -INT32_C(   477063257),  INT32_C(   958464539), -INT32_C(   849722574),  INT32_C(   252794485),  INT32_C(    53279558),  INT32_C(  1053914500) },
      {  INT32_C(  1492769416), -INT32_C(   993371108),  INT32_C(   262626036), -INT32_C(  1907832484), -INT32_C(  2107858420),  INT32_C(  1066495481), -INT32_C(   314393239), -INT32_C(  1171581902) },
       INT32_C(          46),
      {  INT32_C(    82728510), -INT32_C(   793497294),  INT32_C(   636077033),  INT32_C(  1082581906), -INT32_C(  1865637737),  INT32_C(  1436372964),  INT32_C(  1053932368),  INT32_C(  1566649994) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    int       imm8 = test_vec[i].imm8;
    easysimd__m256i r = easysimd_mm256_loadu_si256(test_vec[i].r);
    easysimd__m256i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      EASYSIMD_CONSTIFY_64_(easysimd_mm256_shldi_epi32, ret, easysimd_mm256_setzero_si256(), imm8, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_shldi_epi32");
    easysimd_assert_m256i_i32(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    int       imm8 = easysimd_test_codegen_random_i32() & 63;
    easysimd__m256i r;
    EASYSIMD_CONSTIFY_64_(easysimd_mm256_shldi_epi32, r, easysimd_mm256_setzero_si256(), imm8, a, b);

    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_shldi_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t a[4];
    int64_t b[4];
    int     imm8;
    int64_t r[4];
  } test_vec[8] = {
    { { -INT64_C( 3017953648206724231),  INT64_C( 2489852941423190412),  INT64_C( 1517541379535218607), -INT64_C(  633805237152872282) },
      { -INT64_C( 2412618427294662776),  INT64_C( 7816666469610160457), -INT64_C( 6091779279428214137), -INT64_C(  313687971534852817) },
       INT32_C(          13),
      { -INT64_C( 4439227338685727792), -INT64_C( 5223649383988228721), -INT64_C( 1406524527726955154), -INT64_C( 8597418043945721996) } },
    { { -INT64_C( 1478922106057183785), -INT64_C( 7271078547549523152),  INT64_C( 3203643429716659351),  INT64_C( 2385449072617593390) },
      {  INT64_C( 1736625737194150143), -INT64_C(  513438527056285265), -INT64_C(  767805710707775886),  INT64_C( 5889031912138244175) },
       INT32_C(          35),
      {  INT64_C( 1146569319930257217), -INT64_C( 4987849253851286774),  INT64_C( 4194087927355123249), -INT64_C( 2792001479033988460) } },
    { {  INT64_C( 7614727187880712120),  INT64_C( 8680443852137597913), -INT64_C( 6755859790054961066), -INT64_C( 5866375065662916504) },
      { -INT64_C( 8828249699747776247),  INT64_C( 6158189804804693325), -INT64_C( 6265852602970610920),  INT64_C( 8850922850227715632) },
       INT32_C(           5),
      {  INT64_C( 3863597053958616848),  INT64_C( 1073042162759858986),  INT64_C( 5173415602755865301), -INT64_C( 3256561364117811953) } },
    { { -INT64_C( 1948221466512848981), -INT64_C( 7976099992496935474), -INT64_C( 3908719475717088661),  INT64_C( 3094482566573692219) },
      { -INT64_C(  243449265867544618),  INT64_C( 8531843045507979397),  INT64_C( 5669419584018621068), -INT64_C( 7633037594207522711) },
       INT32_C(          39),
      {  INT64_C( 6558320770614564701),  INT64_C( 2069095320278074859), -INT64_C( 7084947645949965966),  INT64_C( 2434932043902115828) } },
    { { -INT64_C( 3050988869113417396), -INT64_C( 4298864298281988121), -INT64_C( 5367690467136633551),  INT64_C( 1485797313246630945) },
      {  INT64_C(  877232318470058125),  INT64_C(  438211275428689822),  INT64_C( 7603802717011584987), -INT64_C(  318964554414304842) },
       INT32_C(          63),
      {  INT64_C(  438616159235029062), -INT64_C( 9004266399140430897), -INT64_C( 5421470678348983315), -INT64_C(  159482277207152421) } },
    { { -INT64_C( 3821884870432358905),  INT64_C( 8751180384404769478),  INT64_C( 4018921472706934930), -INT64_C( 2772101007571527197) },
      { -INT64_C( 5926045620027525374), -INT64_C( 7698691341419638146),  INT64_C(  124592390674451129),  INT64_C( 8262322392876182781) },
       INT32_C(          34),
      { -INT64_C( 7655565080749550689), -INT64_C( 8700907087990684658),  INT64_C( 2561389612071817338), -INT64_C( 8590220955544540733) } },
    { { -INT64_C( 4482624527058719844),  INT64_C( 1982205051511698245), -INT64_C( 7447194389675492135),  INT64_C( 1375132218519343082) },
      {  INT64_C( 1521928397409635231), -INT64_C( 9077683817391295252),  INT64_C( 5517752190247641554), -INT64_C( 5775604766313243571) },
       INT32_C(          57),
      {  INT64_C( 4047115331728727191), -INT64_C( 8429600313223010071), -INT64_C( 5577384895972069309), -INT64_C( 3071540861829795528) } },
    { {  INT64_C( 5176100104374790911),  INT64_C( 6743897431073217226), -INT64_C( 3012202511497352928),  INT64_C( 1203183555505883759) },
      { -INT64_C( 3485274815693270597), -INT64_C( 5979933077107152213), -INT64_C( 7077321950387000276), -INT64_C( 7387598980080317096) },
       INT32_C(          10),
      {  INT64_C( 6110957725144579902),  INT64_C( 6668685851602135732), -INT64_C( 3889111463794277769), -INT64_C( 3871892100514988443) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    int       imm8 = test_vec[i].imm8;
    easysimd__m256i r = easysimd_mm256_loadu_si256(test_vec[i].r);
    easysimd__m256i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      EASYSIMD_CONSTIFY_64_(easysimd_mm256_shldi_epi64, ret, easysimd_mm256_setzero_si256(), imm8, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_shldi_epi64");
    easysimd_assert_m256i_i64(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    int       imm8 = easysimd_test_codegen_random_i32() & 63;
    easysimd__m256i r;
    EASYSIMD_CONSTIFY_64_(easysimd_mm256_shldi_epi64, r, easysimd_mm256_setzero_si256(), imm8, a, b);

    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_shldi_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int16_t a[32];
    int16_t b[32];
    int     imm8;
    int16_t r[32];
  } test_vec[8] = {
    { { -INT16_C(  9641), -INT16_C(  2766),  INT16_C( 29502),  INT16_C( 30235),  INT16_C(  8121),  INT16_C(  2029), -INT16_C( 31545),  INT16_C(  8960),
        -INT16_C( 19062), -INT16_C( 29942), -INT16_C( 31052), -INT16_C(  8356), -INT16_C(  1365),  INT16_C( 30665), -INT16_C(  5132),  INT16_C( 19446),
         INT16_C( 10438),  INT16_C(  1088),  INT16_C( 23708),  INT16_C( 21882),  INT16_C( 26491),  INT16_C( 16988),  INT16_C( 24043),  INT16_C( 30309),
         INT16_C( 28434), -INT16_C( 14847),  INT16_C( 24309), -INT16_C( 24154),  INT16_C( 28504),  INT16_C( 19480),  INT16_C(  3674),  INT16_C(  8344) },
      { -INT16_C( 10185), -INT16_C( 11484), -INT16_C( 24780), -INT16_C( 20696), -INT16_C( 31738), -INT16_C(  3343),  INT16_C( 22241), -INT16_C(  3224),
         INT16_C( 27078), -INT16_C( 17478),  INT16_C( 24775),  INT16_C(  8284),  INT16_C( 30159),  INT16_C( 10604),  INT16_C(  1155), -INT16_C( 17846),
         INT16_C( 28381),  INT16_C(  4493), -INT16_C( 19187),  INT16_C(  5313), -INT16_C( 19910),  INT16_C(  6918),  INT16_C( 28169), -INT16_C( 12529),
        -INT16_C( 13865), -INT16_C( 24694), -INT16_C(  6359), -INT16_C(  1857),  INT16_C( 11100), -INT16_C(  8415),  INT16_C( 27440),  INT16_C(  3482) },
       INT32_C(          26),
      {  INT16_C( 24416), -INT16_C( 13492), -INT16_C(  1412),  INT16_C( 28348), -INT16_C(  6640), -INT16_C( 18485),  INT16_C(  7515),  INT16_C(   973),
         INT16_C( 10663),  INT16_C( 10990), -INT16_C( 11901),  INT16_C( 28801), -INT16_C( 21033),  INT16_C(  9381), -INT16_C( 12270), -INT16_C(  9495),
         INT16_C(  6587),  INT16_C(    70),  INT16_C( 29396), -INT16_C(  6061), -INT16_C(  4408),  INT16_C( 28780), -INT16_C( 21064), -INT16_C( 26820),
         INT16_C( 19239),  INT16_C(  1662), -INT16_C( 10340), -INT16_C( 25630),  INT16_C( 24749),  INT16_C( 25468),  INT16_C( 27052),  INT16_C( 24630) } },
    { { -INT16_C(  8227),  INT16_C(  6139),  INT16_C(   402), -INT16_C( 25806),  INT16_C( 16751),  INT16_C( 18282), -INT16_C(  3062),  INT16_C( 13286),
        -INT16_C( 23077),  INT16_C( 14123),  INT16_C( 19920),  INT16_C(    23), -INT16_C( 20040), -INT16_C( 28147),  INT16_C( 11480), -INT16_C( 19078),
         INT16_C( 29963), -INT16_C( 25140), -INT16_C(   137), -INT16_C(  6600), -INT16_C( 24000),  INT16_C( 19245),  INT16_C(  5015),  INT16_C( 29310),
        -INT16_C( 21832), -INT16_C( 30294), -INT16_C( 15881), -INT16_C( 20599), -INT16_C( 26766),  INT16_C( 19010), -INT16_C( 17213), -INT16_C( 12800) },
      { -INT16_C( 13263), -INT16_C( 22420), -INT16_C( 23349),  INT16_C(  3215), -INT16_C( 17337), -INT16_C(  8617), -INT16_C( 10800), -INT16_C( 30640),
        -INT16_C(  1409),  INT16_C( 30225), -INT16_C( 25669),  INT16_C( 11558),  INT16_C( 26674), -INT16_C(  2696),  INT16_C( 30756),  INT16_C( 21955),
         INT16_C( 12100),  INT16_C(  4350), -INT16_C( 29228),  INT16_C(  6940),  INT16_C( 29513),  INT16_C(  6649),  INT16_C( 18760), -INT16_C( 14174),
        -INT16_C( 19644), -INT16_C(   194),  INT16_C( 25678), -INT16_C( 32723), -INT16_C( 23092), -INT16_C(  3979),  INT16_C( 14621),  INT16_C( 24902) },
       INT32_C(          40),
      { -INT16_C(  8756), -INT16_C(  1112), -INT16_C( 27996),  INT16_C( 12812),  INT16_C( 28604),  INT16_C( 27358),  INT16_C(  2773), -INT16_C(  6520),
        -INT16_C(  9222),  INT16_C( 11126), -INT16_C( 12133),  INT16_C(  5933), -INT16_C( 18328),  INT16_C(  3573), -INT16_C( 10120),  INT16_C( 31317),
         INT16_C(  2863), -INT16_C( 13296),  INT16_C( 30605),  INT16_C( 14363),  INT16_C( 16499),  INT16_C( 11545), -INT16_C( 26807),  INT16_C( 32456),
        -INT16_C( 18253), -INT16_C( 21761), -INT16_C(  2204), -INT16_C( 30336),  INT16_C( 29349),  INT16_C( 17136), -INT16_C( 15559),  INT16_C(    97) } },
    { { -INT16_C( 29231),  INT16_C(  6743),  INT16_C( 20480),  INT16_C( 18740), -INT16_C( 10598), -INT16_C(  8687),  INT16_C( 20361), -INT16_C( 10019),
         INT16_C(  2740), -INT16_C( 32680), -INT16_C( 12625), -INT16_C( 13199), -INT16_C( 18681),  INT16_C( 28462), -INT16_C( 24581), -INT16_C( 13140),
         INT16_C(   813),  INT16_C( 11750),  INT16_C(  6740), -INT16_C(  4490), -INT16_C( 30736),  INT16_C( 31436), -INT16_C( 22057), -INT16_C( 29870),
        -INT16_C( 21836),  INT16_C( 25355),  INT16_C( 31864),  INT16_C( 32560),  INT16_C( 24115),  INT16_C( 12015), -INT16_C( 25603),  INT16_C( 11002) },
      { -INT16_C(  7778), -INT16_C(  3496), -INT16_C( 12549), -INT16_C(  4896), -INT16_C( 21418),  INT16_C( 11622), -INT16_C( 18346),  INT16_C(  2744),
        -INT16_C( 15518), -INT16_C(  9363), -INT16_C( 25280),  INT16_C( 29530),  INT16_C( 18939), -INT16_C(  1630), -INT16_C( 25372), -INT16_C( 31965),
         INT16_C( 31613),  INT16_C( 31093),  INT16_C( 22090), -INT16_C( 24475), -INT16_C( 13566),  INT16_C( 22733), -INT16_C( 31357), -INT16_C(  6814),
        -INT16_C( 12216), -INT16_C( 30528),  INT16_C(  7021),  INT16_C( 27132), -INT16_C( 24988),  INT16_C( 18786), -INT16_C( 31430), -INT16_C( 18228) },
       INT32_C(           1),
      {  INT16_C(  7075),  INT16_C( 13487), -INT16_C( 24575), -INT16_C( 28055), -INT16_C( 21195), -INT16_C( 17374), -INT16_C( 24813), -INT16_C( 20038),
         INT16_C(  5481),  INT16_C(   177), -INT16_C( 25249), -INT16_C( 26398),  INT16_C( 28174), -INT16_C(  8611),  INT16_C( 16375), -INT16_C( 26279),
         INT16_C(  1626),  INT16_C( 23500),  INT16_C( 13480), -INT16_C(  8979),  INT16_C(  4065), -INT16_C(  2664),  INT16_C( 21423),  INT16_C(  5797),
         INT16_C( 21865), -INT16_C( 14825), -INT16_C(  1808), -INT16_C(   416), -INT16_C( 17305),  INT16_C( 24030),  INT16_C( 14331),  INT16_C( 22005) } },
    { { -INT16_C( 26985), -INT16_C( 25877), -INT16_C( 18335), -INT16_C(  6926),  INT16_C( 21821), -INT16_C( 31287), -INT16_C( 30171), -INT16_C( 28146),
         INT16_C(  2725),  INT16_C(  2555),  INT16_C( 23976), -INT16_C(  7598),  INT16_C(  7907), -INT16_C(  7014), -INT16_C( 13472), -INT16_C(  2257),
         INT16_C(  6753), -INT16_C( 15727), -INT16_C( 31534),  INT16_C(  4006),  INT16_C( 28889), -INT16_C(   364), -INT16_C( 23814), -INT16_C( 24688),
        -INT16_C( 29524),  INT16_C( 21672), -INT16_C(  1047), -INT16_C( 13257), -INT16_C( 12007),  INT16_C( 31152), -INT16_C(  8291), -INT16_C(   399) },
      {  INT16_C(   761), -INT16_C( 13375),  INT16_C( 26502),  INT16_C( 24538),  INT16_C( 28631), -INT16_C( 11939), -INT16_C(  4591), -INT16_C( 16784),
         INT16_C(  6522),  INT16_C( 25362),  INT16_C( 18708),  INT16_C( 11568), -INT16_C(  8165), -INT16_C( 18265),  INT16_C(  6336), -INT16_C( 17994),
         INT16_C( 30490), -INT16_C( 24187),  INT16_C( 24543), -INT16_C( 18944),  INT16_C( 24270), -INT16_C(  8056), -INT16_C(  1972), -INT16_C( 14690),
        -INT16_C( 20463),  INT16_C(  9513),  INT16_C( 23034),  INT16_C(  5459), -INT16_C(  1478), -INT16_C(  1331), -INT16_C( 31982),  INT16_C( 11443) },
       INT32_C(          59),
      { -INT16_C( 18409),  INT16_C( 24158),  INT16_C(  2876), -INT16_C( 27906), -INT16_C(  5250),  INT16_C( 20106),  INT16_C( 12144),  INT16_C( 30195),
         INT16_C( 10443), -INT16_C(  9448),  INT16_C( 16968), -INT16_C( 28311),  INT16_C(  7936), -INT16_C( 10811),  INT16_C(   198),  INT16_C( 32205),
         INT16_C(  3000), -INT16_C( 29428), -INT16_C( 27906),  INT16_C( 13744), -INT16_C( 13578), -INT16_C( 22780), -INT16_C( 10302), -INT16_C( 31180),
         INT16_C( 25984),  INT16_C( 16681),  INT16_C( 19151), -INT16_C( 18262), -INT16_C( 12335), -INT16_C( 30762), -INT16_C(  5096), -INT16_C( 30363) } },
    { { -INT16_C( 12648),  INT16_C( 26256),  INT16_C(  6188),  INT16_C( 30790), -INT16_C(  7151),  INT16_C(  8766),  INT16_C( 26517), -INT16_C( 28856),
        -INT16_C( 25663), -INT16_C(  1116),  INT16_C( 29077), -INT16_C( 22539), -INT16_C( 22284), -INT16_C(  4141), -INT16_C( 24095),  INT16_C( 31177),
         INT16_C( 23151), -INT16_C( 25633),  INT16_C(  9842), -INT16_C( 31981),  INT16_C( 20746), -INT16_C( 24666), -INT16_C(  4424),  INT16_C( 31022),
        -INT16_C( 11639),  INT16_C(  7796),  INT16_C( 26947),  INT16_C( 14533), -INT16_C( 26606), -INT16_C(  3289), -INT16_C(  3783), -INT16_C( 22420) },
      {  INT16_C( 19275), -INT16_C( 17085),  INT16_C( 22129),  INT16_C( 31809), -INT16_C(  6233),  INT16_C( 24603),  INT16_C( 19157),  INT16_C( 24281),
         INT16_C( 19996),  INT16_C( 24700),  INT16_C( 16823), -INT16_C( 13928), -INT16_C( 16423),  INT16_C(  5052),  INT16_C( 10416), -INT16_C(  1093),
        -INT16_C(   140), -INT16_C(  6727), -INT16_C(  1451), -INT16_C(   671),  INT16_C( 32225), -INT16_C( 18851),  INT16_C( 14023), -INT16_C(  7404),
        -INT16_C( 28540),  INT16_C( 15427), -INT16_C(  9263), -INT16_C( 22011), -INT16_C( 15717),  INT16_C( 19389),  INT16_C( 31210),  INT16_C( 24135) },
       INT32_C(          56),
      { -INT16_C( 26549), -INT16_C( 28483),  INT16_C( 11350),  INT16_C( 18044),  INT16_C(  4583),  INT16_C( 15968), -INT16_C( 27318),  INT16_C( 18526),
        -INT16_C( 16050), -INT16_C( 23456), -INT16_C( 27327), -INT16_C(  2615), -INT16_C(  2881), -INT16_C( 11501), -INT16_C(  7896), -INT16_C( 13829),
         INT16_C( 28671), -INT16_C(  8219),  INT16_C( 29434),  INT16_C(  5117),  INT16_C(  2685), -INT16_C( 22858), -INT16_C( 18378),  INT16_C( 12003),
        -INT16_C( 30320),  INT16_C( 29756),  INT16_C( 17371), -INT16_C( 14934),  INT16_C(  4802),  INT16_C( 10059),  INT16_C( 14713),  INT16_C( 27742) } },
    { { -INT16_C( 23046), -INT16_C(  9270),  INT16_C( 10018), -INT16_C(  5743), -INT16_C( 23202), -INT16_C(  7475),  INT16_C(  4149),  INT16_C(  1566),
         INT16_C(  9452), -INT16_C( 30800),  INT16_C( 28390), -INT16_C( 12078),  INT16_C(  6631),  INT16_C( 24367),  INT16_C( 29465),  INT16_C(  4908),
        -INT16_C(  2280),  INT16_C( 15342),  INT16_C( 32542),  INT16_C( 31780), -INT16_C(  3804),  INT16_C( 22879),  INT16_C( 32002), -INT16_C(  4513),
         INT16_C(  4257), -INT16_C( 30859),  INT16_C( 18302),  INT16_C( 25944), -INT16_C( 30879),  INT16_C( 31428), -INT16_C(  3846),  INT16_C(  4750) },
      {  INT16_C( 31975),  INT16_C(  1613),  INT16_C( 29436),  INT16_C(  8322), -INT16_C(  7837),  INT16_C( 25978), -INT16_C(  9889),  INT16_C(    83),
        -INT16_C( 14103),  INT16_C( 26504), -INT16_C(  8176),  INT16_C( 29132), -INT16_C( 28569),  INT16_C( 25067),  INT16_C( 31105),  INT16_C( 26739),
        -INT16_C( 15882), -INT16_C(  3474), -INT16_C(  3789), -INT16_C( 27118), -INT16_C( 29486),  INT16_C( 12796),  INT16_C( 20326),  INT16_C( 20274),
        -INT16_C( 17896),  INT16_C( 10423), -INT16_C( 31846),  INT16_C(   409), -INT16_C( 31724), -INT16_C( 27294), -INT16_C( 10754), -INT16_C(  2819) },
       INT32_C(          22),
      {  INT16_C( 32415), -INT16_C(  3455), -INT16_C( 14180),  INT16_C( 25672),  INT16_C( 22456), -INT16_C( 19623),  INT16_C(  3446), -INT16_C( 30848),
         INT16_C( 15154), -INT16_C(  5095), -INT16_C( 17992),  INT16_C( 13468),  INT16_C( 31204), -INT16_C( 13352), -INT16_C( 14754), -INT16_C( 13542),
        -INT16_C( 14800), -INT16_C(  1092), -INT16_C( 14404),  INT16_C(  2341),  INT16_C( 18723),  INT16_C( 22476),  INT16_C( 16531), -INT16_C( 26669),
         INT16_C( 10350), -INT16_C(  8886), -INT16_C(  8288),  INT16_C( 22016), -INT16_C( 10143), -INT16_C( 20187),  INT16_C( 16053), -INT16_C( 23619) } },
    { { -INT16_C(  1955),  INT16_C( 12128),  INT16_C( 23685), -INT16_C(  5279), -INT16_C( 27733), -INT16_C( 15558), -INT16_C(  3763), -INT16_C(  6165),
        -INT16_C( 31627), -INT16_C( 30232),  INT16_C( 18953),  INT16_C(  1822),  INT16_C(  6943), -INT16_C( 18693), -INT16_C(  7801), -INT16_C(  7041),
        -INT16_C(  8231),  INT16_C( 24084),  INT16_C( 30011), -INT16_C(  6327), -INT16_C( 31736),  INT16_C( 21930), -INT16_C( 27019), -INT16_C(  5572),
         INT16_C(  9242),  INT16_C(  9075), -INT16_C( 28306), -INT16_C( 29398),  INT16_C(  9645),  INT16_C( 13379), -INT16_C( 15610), -INT16_C(  8167) },
      {  INT16_C( 11682), -INT16_C(  8642), -INT16_C( 30558), -INT16_C( 21819),  INT16_C( 28428), -INT16_C( 32257),  INT16_C( 15109),  INT16_C(  8300),
        -INT16_C(  8353), -INT16_C( 12989),  INT16_C( 28273),  INT16_C(  7770), -INT16_C( 24941), -INT16_C( 26030),  INT16_C( 27489),  INT16_C(   890),
        -INT16_C( 18280),  INT16_C( 15073), -INT16_C( 22976),  INT16_C( 19684), -INT16_C(  7402),  INT16_C(  7118),  INT16_C( 14878),  INT16_C( 32059),
         INT16_C( 32537), -INT16_C( 30134), -INT16_C( 23059), -INT16_C( 32600), -INT16_C(  1213), -INT16_C( 23526), -INT16_C( 27546), -INT16_C(    89) },
       INT32_C(          13),
      { -INT16_C( 23116),  INT16_C(  7111), -INT16_C( 20204),  INT16_C( 13656),  INT16_C( 28129),  INT16_C( 20543), -INT16_C( 22688),  INT16_C( 25613),
        -INT16_C( 17429),  INT16_C(  6568),  INT16_C( 11726), -INT16_C( 15413), -INT16_C(  3118),  INT16_C( 29514), -INT16_C(  4756), -INT16_C(  8081),
         INT16_C( 14099), -INT16_C( 30884),  INT16_C( 29896),  INT16_C( 10652),  INT16_C(  7266),  INT16_C( 17273), -INT16_C( 22717), -INT16_C( 28761),
         INT16_C( 20451),  INT16_C( 29001), -INT16_C( 11075),  INT16_C( 20501), -INT16_C( 16536),  INT16_C( 29827), -INT16_C( 11636),  INT16_C( 16372) } },
    { {  INT16_C(  7727),  INT16_C( 17882), -INT16_C( 22527),  INT16_C(  8289), -INT16_C( 25374), -INT16_C(  1123), -INT16_C(  6117),  INT16_C(  2182),
         INT16_C( 11917), -INT16_C( 12151), -INT16_C( 23767), -INT16_C( 28556),  INT16_C(  6968), -INT16_C( 31345), -INT16_C( 14172), -INT16_C( 11246),
        -INT16_C(  4890), -INT16_C(  6119),  INT16_C( 31380),  INT16_C( 30216), -INT16_C( 23273),  INT16_C( 12914), -INT16_C(  1907),  INT16_C(  6715),
        -INT16_C( 15322),  INT16_C( 20714),  INT16_C( 24167), -INT16_C( 24608),  INT16_C( 28538),  INT16_C(  7716),  INT16_C( 14135),  INT16_C(  7922) },
      {  INT16_C(  3107), -INT16_C( 18426),  INT16_C(  3718), -INT16_C( 25298), -INT16_C( 24397),  INT16_C( 16848),  INT16_C(  2968), -INT16_C( 16549),
         INT16_C( 18127),  INT16_C( 13839), -INT16_C(  4188),  INT16_C(  7894), -INT16_C(  1442), -INT16_C( 27331),  INT16_C( 12081),  INT16_C( 21939),
        -INT16_C( 18117), -INT16_C( 15859),  INT16_C( 15303),  INT16_C( 31583),  INT16_C( 12252),  INT16_C( 29884),  INT16_C(  5946),  INT16_C(  2355),
         INT16_C( 16989),  INT16_C(   576),  INT16_C(  5681), -INT16_C( 28896),  INT16_C( 23824),  INT16_C( 16933), -INT16_C( 10099), -INT16_C( 14185) },
       INT32_C(          18),
      {  INT16_C( 30908),  INT16_C(  5994), -INT16_C( 24572), -INT16_C( 32378),  INT16_C( 29578), -INT16_C(  4491), -INT16_C( 24468),  INT16_C(  8730),
        -INT16_C( 17867),  INT16_C( 16932), -INT16_C( 29529),  INT16_C( 16848),  INT16_C( 27875),  INT16_C(  5694),  INT16_C(  8848),  INT16_C( 20553),
        -INT16_C( 19558), -INT16_C( 24473), -INT16_C(  5552), -INT16_C( 10207), -INT16_C( 27556), -INT16_C( 13879), -INT16_C(  7628),  INT16_C( 26860),
         INT16_C(  4249),  INT16_C( 17320),  INT16_C( 31132),  INT16_C( 32642), -INT16_C( 16919),  INT16_C( 30865), -INT16_C(  8993),  INT16_C( 31691) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    int       imm8 = test_vec[i].imm8;
    easysimd__m512i r = easysimd_mm512_loadu_si512(test_vec[i].r);
    easysimd__m512i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      EASYSIMD_CONSTIFY_64_(easysimd_mm512_shldi_epi16, ret, easysimd_mm512_setzero_si512(), imm8, a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_shldi_epi16");
    easysimd_assert_m512i_i16(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i16x32();
    easysimd__m512i b = easysimd_test_x86_random_i16x32();
    int       imm8 = easysimd_test_codegen_random_i32() & 63;
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_64_(easysimd_mm512_shldi_epi16, r, easysimd_mm512_setzero_si512(), imm8, a, b);

    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_shldi_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int32_t a[16];
    int32_t b[16];
    int     imm8;
    int32_t r[16];
  } test_vec[8] = {
    { { -INT32_C(  1223484011),  INT32_C(   312182796),  INT32_C(   860626031),  INT32_C(  1363454574), -INT32_C(   384011114), -INT32_C(   412332852),  INT32_C(  1846871780),  INT32_C(    46290797),
        -INT32_C(  2135304844), -INT32_C(   862825379), -INT32_C(   553590927),  INT32_C(   523256969), -INT32_C(  2062988103),  INT32_C(  2104259993), -INT32_C(   370376068),  INT32_C(  1324068569) },
      { -INT32_C(   523328381),  INT32_C(  1772970488), -INT32_C(   917918400), -INT32_C(  1427539471),  INT32_C(  1596977862), -INT32_C(   472081305), -INT32_C(   137574370), -INT32_C(   112871562),
         INT32_C(  1423578203), -INT32_C(  1245870219),  INT32_C(   629016116),  INT32_C(  1188063104), -INT32_C(  1046085799), -INT32_C(  1180401253), -INT32_C(  1062112950), -INT32_C(  2068122072) },
       INT32_C(          10),
      {  INT32_C(  1282824067),  INT32_C(  1847603622),  INT32_C(   812760869),  INT32_C(   313113259),  INT32_C(  1909610876), -INT32_C(  1322044529),  INT32_C(  1411093471),  INT32_C(   157136869),
        -INT32_C(   413806253),  INT32_C(  1230075606),  INT32_C(    58573973), -INT32_C(  1055775461),  INT32_C(   624092934), -INT32_C(  1311349018), -INT32_C(  1307970814), -INT32_C(  1363450350) } },
    { {  INT32_C(  1328846107),  INT32_C(   477410204),  INT32_C(  1952597018), -INT32_C(   550172860), -INT32_C(   761669240),  INT32_C(  1939032650),  INT32_C(  1274498368), -INT32_C(    53817375),
         INT32_C(     4980580), -INT32_C(   870531150),  INT32_C(  1212186116),  INT32_C(   237467013), -INT32_C(  1713323953),  INT32_C(  1259107082), -INT32_C(  1584004160),  INT32_C(   933060818) },
      {  INT32_C(   288876639), -INT32_C(  1394715480),  INT32_C(  1492393682), -INT32_C(   479847532), -INT32_C(   428063012),  INT32_C(  2050066874),  INT32_C(  1595721612), -INT32_C(  2020165336),
         INT32_C(  1251593634), -INT32_C(   185174239),  INT32_C(   692906645), -INT32_C(   519261691), -INT32_C(  1295480328), -INT32_C(  1641154286), -INT32_C(   369276479), -INT32_C(  1536126206) },
       INT32_C(          33),
      { -INT32_C(  1637275082),  INT32_C(   954820409), -INT32_C(   389773260), -INT32_C(  1100345719), -INT32_C(  1523338479), -INT32_C(   416901996), -INT32_C(  1745970560), -INT32_C(   107634749),
         INT32_C(     9961160), -INT32_C(  1741062299), -INT32_C(  1870595064),  INT32_C(   474934027),  INT32_C(   868319391), -INT32_C(  1776753131),  INT32_C(  1126958977),  INT32_C(  1866121637) } },
    { {  INT32_C(   376890496), -INT32_C(   734018866),  INT32_C(  1840598132), -INT32_C(   417366571),  INT32_C(   948259959), -INT32_C(  1742634091),  INT32_C(  2000458006),  INT32_C(   469379995),
        -INT32_C(   567185393), -INT32_C(  1481477837), -INT32_C(  1844156483),  INT32_C(  1551447269),  INT32_C(   378863232), -INT32_C(  1733380735), -INT32_C(   519050682),  INT32_C(   637339925) },
      { -INT32_C(  1392300422),  INT32_C(  1549055391),  INT32_C(    32401692),  INT32_C(   492660637), -INT32_C(   416026266), -INT32_C(   310386266), -INT32_C(   506556468),  INT32_C(   302435224),
        -INT32_C(  1715533318), -INT32_C(   604629826),  INT32_C(   450684029), -INT32_C(  1305003444), -INT32_C(   761697493),  INT32_C(   415177036),  INT32_C(  1090162088),  INT32_C(  1397948505) },
       INT32_C(          10),
      { -INT32_C(   611188044), -INT32_C(    16041615), -INT32_C(   718155769),  INT32_C(  2113361013),  INT32_C(   355590044), -INT32_C(  2045880395), -INT32_C(   230401145), -INT32_C(   391222200),
        -INT32_C(   977256858), -INT32_C(   909848721),  INT32_C(  1369371755), -INT32_C(   455895352),  INT32_C(  1408893770), -INT32_C(  1160379294),  INT32_C(  1068046595), -INT32_C(   198945459) } },
    { { -INT32_C(  1549540826),  INT32_C(   331186375), -INT32_C(   440011334), -INT32_C(  1397268896),  INT32_C(   533034615),  INT32_C(  1566621444), -INT32_C(   911166529), -INT32_C(   325935931),
         INT32_C(  1183790463),  INT32_C(  1868123573), -INT32_C(  1571479998), -INT32_C(   196211588), -INT32_C(  2028792957), -INT32_C(  1830521902), -INT32_C(   329542618), -INT32_C(  1344738000) },
      { -INT32_C(   655005917), -INT32_C(   163033420), -INT32_C(   375874196),  INT32_C(   752740265), -INT32_C(   843845382), -INT32_C(  1973446812),  INT32_C(  1551284779), -INT32_C(   888451416),
         INT32_C(  1789067702), -INT32_C(  1134433457),  INT32_C(   849738120), -INT32_C(   614563104), -INT32_C(   693628302), -INT32_C(   715061335),  INT32_C(  1764874177), -INT32_C(   600556506) },
       INT32_C(          61),
      { -INT32_C(   618746652), -INT32_C(    20379178),  INT32_C(  1563628461),  INT32_C(    94092533), -INT32_C(   105480673), -INT32_C(  1857293588), -INT32_C(   342960315), -INT32_C(  1184798251),
        -INT32_C(   313237450), -INT32_C(  1215546007),  INT32_C(  1179959089), -INT32_C(  1687433124),  INT32_C(  2060780110),  INT32_C(  1521230069), -INT32_C(   853132552),  INT32_C(   461801348) } },
    { {  INT32_C(  1279895491), -INT32_C(  2122387807), -INT32_C(   497230736), -INT32_C(  1732705042), -INT32_C(   848487925),  INT32_C(   389455601),  INT32_C(   418606042),  INT32_C(   111491651),
        -INT32_C(  2108494111),  INT32_C(  1275384028), -INT32_C(  1691459411),  INT32_C(  1882449765), -INT32_C(   214065151),  INT32_C(   420115518),  INT32_C(   573701855),  INT32_C(   422106680) },
      { -INT32_C(  1600357436), -INT32_C(   118710197),  INT32_C(  1704204800),  INT32_C(    64407298), -INT32_C(  1493822616),  INT32_C(  1723793799),  INT32_C(   931721471), -INT32_C(  1974423098),
         INT32_C(  1999301676), -INT32_C(  1922034036),  INT32_C(   871498801),  INT32_C(   859228363),  INT32_C(  1675242972),  INT32_C(   768252206),  INT32_C(  1348752010),  INT32_C(   802927619) },
       INT32_C(          33),
      { -INT32_C(  1735176313),  INT32_C(    50191683), -INT32_C(   994461472),  INT32_C(   829557212), -INT32_C(  1696975849),  INT32_C(   778911202),  INT32_C(   837212084),  INT32_C(   222983303),
         INT32_C(    77979074), -INT32_C(  1744199239),  INT32_C(   912048474), -INT32_C(   530067766), -INT32_C(   428130302),  INT32_C(   840231036),  INT32_C(  1147403710),  INT32_C(   844213360) } },
    { {  INT32_C(  1304041244), -INT32_C(   427774693),  INT32_C(  1360705141),  INT32_C(   297137379),  INT32_C(   389971853), -INT32_C(   714562863), -INT32_C(   133938345),  INT32_C(  1680190280),
        -INT32_C(   575545150),  INT32_C(    46412173), -INT32_C(   883630360),  INT32_C(  1608321490),  INT32_C(  1500978056),  INT32_C(   355393470),  INT32_C(  1779249954), -INT32_C(  1580256546) },
      { -INT32_C(  1619099630), -INT32_C(  1700642126), -INT32_C(   228198880), -INT32_C(  2024652033),  INT32_C(   484493662), -INT32_C(   902754392),  INT32_C(   540294722), -INT32_C(  2084502672),
         INT32_C(   908214404), -INT32_C(  1546599294), -INT32_C(  1181403718), -INT32_C(   700389512),  INT32_C(  1492263344),  INT32_C(  1914839856), -INT32_C(   778938527), -INT32_C(   564898726) },
       INT32_C(          20),
      {  INT32_C(  1909061608), -INT32_C(   776361436),  INT32_C(  1734288991),  INT32_C(  1312322852), -INT32_C(   120467956),  INT32_C(   756851472),  INT32_C(   896664387), -INT32_C(  1266140144),
         INT32_C(   203645476),  INT32_C(   416955660), -INT32_C(   292841133), -INT32_C(  1657969650), -INT32_C(  1199206622), -INT32_C(    68738526),  INT32_C(   841816357),  INT32_C(   770565445) } },
    { { -INT32_C(   155589573), -INT32_C(  1834004710), -INT32_C(   412553162),  INT32_C(  1094670865), -INT32_C(   558669187),  INT32_C(   330253753),  INT32_C(   770835609), -INT32_C(  1237055877),
         INT32_C(    95223275), -INT32_C(  2087232436),  INT32_C(  1567293260), -INT32_C(   694245031), -INT32_C(   994815733),  INT32_C(   819487638), -INT32_C(   497169817), -INT32_C(  1147559472) },
      { -INT32_C(   339720801), -INT32_C(   311535711), -INT32_C(  1354049450), -INT32_C(  1920604030), -INT32_C(   799917767),  INT32_C(    67119773), -INT32_C(   991470092), -INT32_C(  1635745537),
         INT32_C(  1737048261), -INT32_C(   296421224),  INT32_C(  1386127312), -INT32_C(  1042275449), -INT32_C(    91147683),  INT32_C(  1358860636), -INT32_C(   317397522),  INT32_C(   730567781) },
       INT32_C(          21),
      { -INT32_C(  2021820408), -INT32_C(   480399926),  INT32_C(   114682203),  INT32_C(  1110552765),  INT32_C(  1337592391), -INT32_C(  1222606843), -INT32_C(  1825006357), -INT32_C(   814493680),
        -INT32_C(  1116933848), -INT32_C(  1986147681), -INT32_C(   376810541),  INT32_C(   725105668),  INT32_C(   561992230),  INT32_C(  1925849042),  INT32_C(  1291690652),  INT32_C(   973435250) } },
    { { -INT32_C(   581179891),  INT32_C(   221313413),  INT32_C(  2060324893), -INT32_C(  1636540606), -INT32_C(   554798352), -INT32_C(  1110703528),  INT32_C(  1827166103),  INT32_C(  2044295788),
        -INT32_C(   430492575),  INT32_C(  1274251054), -INT32_C(   641351273),  INT32_C(   276248864),  INT32_C(    66020779), -INT32_C(    20858009),  INT32_C(  2137696530), -INT32_C(  2047327452) },
      { -INT32_C(  1502916744),  INT32_C(  1844535254),  INT32_C(  1095153184), -INT32_C(  1705918993), -INT32_C(  1969340382),  INT32_C(   243818491),  INT32_C(   747500296), -INT32_C(  1364097738),
        -INT32_C(  1420550699), -INT32_C(  1676065412), -INT32_C(   354590725),  INT32_C(  1065692957),  INT32_C(  1808343919), -INT32_C(  1971760510),  INT32_C(  2075592261),  INT32_C(  1613392011) },
       INT32_C(           5),
      { -INT32_C(  1417887308), -INT32_C(  1507905363),  INT32_C(  1505887144), -INT32_C(   829691821), -INT32_C(   573678063), -INT32_C(  1182774527), -INT32_C(  1660226843),  INT32_C(   992955797),
        -INT32_C(   890860491),  INT32_C(  2121328083),  INT32_C(   951595773),  INT32_C(   250029063),  INT32_C(  2112664941), -INT32_C(   667456271), -INT32_C(   313187761), -INT32_C(  1089969012) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    int       imm8 = test_vec[i].imm8;
    easysimd__m512i r = easysimd_mm512_loadu_si512(test_vec[i].r);
    easysimd__m512i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      EASYSIMD_CONSTIFY_64_(easysimd_mm512_shldi_epi32, ret, easysimd_mm512_setzero_si512(), imm8, a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_shldi_epi32");
    easysimd_assert_m512i_i32(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    int       imm8 = easysimd_test_codegen_random_i32() & 63;
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_64_(easysimd_mm512_shldi_epi32, r, easysimd_mm512_setzero_si512(), imm8, a, b);

    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_shldi_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    int64_t a[8];
    int64_t b[8];
    int     imm8;
    int64_t r[8];
  } test_vec[8] = {
    { {  INT64_C( 4185160333707131536), -INT64_C( 7095617156165741595), -INT64_C( 4021336967812429581), -INT64_C( 2968645908709662082),
         INT64_C(  854334994328418682), -INT64_C( 5876921833400079096), -INT64_C( 6133815645971819326),  INT64_C(  796788113072804337) },
      {  INT64_C( 7885107387817537339), -INT64_C( 7035554060291743244), -INT64_C( 5018814516747142524),  INT64_C( 5513251543386836711),
        -INT64_C( 6196358566787535056),  INT64_C( 8536210198343795453), -INT64_C( 6218486185104954321), -INT64_C( 2942075795446585907) },
       INT32_C(          20),
      { -INT64_C( 3286314139461953832), -INT64_C( 2547197711818824245),  INT64_C( 8455216159115683225),  INT64_C(  316579196789114928),
         INT64_C( 5938561358993465377), -INT64_C( 6076139613686634645), -INT64_C(  958842458187588809),  INT64_C( 2959866975869563578) } },
    { { -INT64_C( 6716117528663986443),  INT64_C( 3489231355891958084),  INT64_C( 7739712781197909948),  INT64_C( 7061836000858903456),
         INT64_C(  607459029355410716), -INT64_C( 9223117208669157791),  INT64_C( 8518122007777958406),  INT64_C( 4816013926928709638) },
      { -INT64_C( 5986273143586674144),  INT64_C( 6216510443200669698), -INT64_C( 3989811303414673579), -INT64_C( 7550548142861873969),
        -INT64_C( 4381605782077291205),  INT64_C( 9112697567770619344), -INT64_C( 4780770089615386851),  INT64_C( 3379082851624608470) },
       INT32_C(          46),
      {  INT64_C( 6610557729718319363),  INT64_C( 1175744691825799518), -INT64_C( 8651641210333525304), -INT64_C( 6563954891203674802),
         INT64_C( 5568473118511489375), -INT64_C( 2190896087789074398), -INT64_C( 5295758817769048086), -INT64_C( 2449523094646369428) } },
    { { -INT64_C( 4991894310645209655), -INT64_C( 5474721530837979384), -INT64_C( 9057262703483966165),  INT64_C( 6640949709042103052),
         INT64_C( 2203860105382255433), -INT64_C( 2640716113246892321), -INT64_C(  357706387639448585),  INT64_C( 7371934222257666398) },
      { -INT64_C( 3283168163289545669), -INT64_C(  779172220731083611), -INT64_C( 7671736591110697827),  INT64_C( 6395907643374222910),
         INT64_C( 9213830311952581400), -INT64_C( 4125293784918108467), -INT64_C(   82231648827852017),  INT64_C( 5696520500471751320) },
       INT32_C(          31),
      {  INT64_C( 5537583544319601657),  INT64_C( 1377001443828427135), -INT64_C( 9171344702299684325), -INT64_C( 7108903947743434226),
         INT64_C( 7925048523650698477), -INT64_C( 3268936649263808108), -INT64_C( 4364438855878840992), -INT64_C(  532730223561077349) } },
    { { -INT64_C( 5158506614646818351),  INT64_C( 2641028460545190224),  INT64_C( 1944003509001290411), -INT64_C( 7948637010335823721),
        -INT64_C( 4159469570871055874),  INT64_C( 4704798805510768065), -INT64_C( 9201557824079565489),  INT64_C( 8780584940190807944) },
      {  INT64_C( 7155180367925152016), -INT64_C( 8076409607548210413),  INT64_C( 8622721956443222081), -INT64_C( 2829497755968151098),
         INT64_C( 2641830747096138659),  INT64_C( 5182999984161501726), -INT64_C( 8823083017209532633), -INT64_C( 8314063925800382723) },
       INT32_C(           5),
      {  INT64_C(  948484994687777324), -INT64_C( 7720809631101670895),  INT64_C( 6867880066912638318),  INT64_C( 3898032701187363579),
        -INT64_C( 3975817751906926652),  INT64_C( 2979609186668165160),  INT64_C(  698054808806730224),  INT64_C( 4277556980462579985) } },
    { { -INT64_C( 5538222830074037592),  INT64_C( 9096546394901778943),  INT64_C(  237787095610150024), -INT64_C( 8076817908999656883),
        -INT64_C( 6180212286002207781),  INT64_C( 3155069108447360650), -INT64_C( 1947809484727268134), -INT64_C( 4440725258341284120) },
      { -INT64_C( 5181244936824376125), -INT64_C( 3419695608506357572),  INT64_C( 2383455581609737368), -INT64_C( 5986320842304157117),
         INT64_C( 8641056089621280260),  INT64_C( 5972443507637808667), -INT64_C( 6415213413156083811),  INT64_C( 4446667160262076703) },
       INT32_C(          50),
      { -INT64_C( 1539421410748155606),  INT64_C( 5187938049075018767),  INT64_C( 3612032375734976276), -INT64_C( 6253613458631906362),
        -INT64_C( 8111581420592510872), -INT64_C( 2726565045350152357),  INT64_C( 3704945039866026506),  INT64_C( 1990862438322677571) } },
    { {  INT64_C(  229679496970984194),  INT64_C( 2004339023653704029), -INT64_C( 1543166006036084627),  INT64_C( 2157109989358455631),
         INT64_C( 7944184889962337779), -INT64_C( 2788229081925643141), -INT64_C(  964338064696209407),  INT64_C( 5800112254069491014) },
      { -INT64_C( 7009910942560634568),  INT64_C( 6387265732971929439),  INT64_C( 5459954866225613360),  INT64_C( 6060202551728208237),
        -INT64_C( 3411946823250762012), -INT64_C( 4025302621255600269),  INT64_C( 4871313520041078346),  INT64_C( 6181796806784796250) },
       INT32_C(          13),
      { -INT64_C(   33456332071742506),  INT64_C( 1943056169642470164), -INT64_C( 5596230956562405000), -INT64_C(  935789789281916285),
        -INT64_C( 1350473475827009004), -INT64_C( 4103475882443704060), -INT64_C( 4650962443659368333), -INT64_C( 4293148538534573383) } },
    { {  INT64_C( 1294972334604419771),  INT64_C( 2271350544374766571), -INT64_C( 2907514024731992106), -INT64_C( 7880867414258454378),
         INT64_C( 4978369134887743627),  INT64_C( 4268568456103456980), -INT64_C( 6531326601233457377), -INT64_C( 7139707267774314367) },
      {  INT64_C( 3961680456807721562), -INT64_C( 7690017906807769689),  INT64_C( 4964505844706882096), -INT64_C(  888187038826959105),
        -INT64_C( 5996083284031288016), -INT64_C( 2368329704341431016), -INT64_C( 5495794846664868429),  INT64_C( 1044114789732997105) },
       INT32_C(          62),
      { -INT64_C( 3621265904225457514), -INT64_C( 1922504476701942423), -INT64_C( 7982245575678055284), -INT64_C( 4833732778134127681),
        -INT64_C( 1499020821007822004),  INT64_C( 4019603592342030150), -INT64_C( 1373948711666217108),  INT64_C( 4872714715860637180) } },
    { {  INT64_C( 3993893012876782535), -INT64_C( 9200233034294063319),  INT64_C( 4878515934739041592),  INT64_C( 2155521443714117736),
        -INT64_C( 6188674306290198178), -INT64_C( 6897586735257603079), -INT64_C( 2698081196897109274),  INT64_C( 3540087499122663597) },
      {  INT64_C( 9182617018180246595), -INT64_C( 5372641730295957912),  INT64_C( 7228716895027935748), -INT64_C( 9190853721651195281),
         INT64_C( 2930071770017139620),  INT64_C( 8948486950771879338),  INT64_C( 3440544228989677731),  INT64_C( 5932056198218065558) },
       INT32_C(          48),
      {  INT64_C( 7766316198039831831), -INT64_C( 8130768157270880188),  INT64_C(  952621622655153665),  INT64_C( 3488179145076916839),
         INT64_C( 6439629226533464645),  INT64_C( 3745161108210088441),  INT64_C( 8855818215809309427), -INT64_C( 2257057322248514959) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    int       imm8 = test_vec[i].imm8;
    easysimd__m512i r = easysimd_mm512_loadu_si512(test_vec[i].r);
    easysimd__m512i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      EASYSIMD_CONSTIFY_64_(easysimd_mm512_shldi_epi64, ret, easysimd_mm512_setzero_si512(), imm8, a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_shldi_epi64");
    easysimd_assert_m512i_i64(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i b = easysimd_test_x86_random_i64x8();
    int       imm8 = easysimd_test_codegen_random_i32() & 63;
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_64_(easysimd_mm512_shldi_epi64, r, easysimd_mm512_setzero_si512(), imm8, a, b);

    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_shldi_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    int16_t    a[8];
    int16_t    b[8];
    int        imm8;
    int16_t    r[8];
  } test_vec[8] = {
    { UINT8_C(117),
      { -INT16_C(  8235), -INT16_C( 30032), -INT16_C(  8762),  INT16_C( 20822), -INT16_C( 22219), -INT16_C( 13900),  INT16_C( 21666), -INT16_C( 19922) },
      {  INT16_C( 13067), -INT16_C( 29400),  INT16_C( 30942), -INT16_C( 26155), -INT16_C( 15191),  INT16_C( 12037), -INT16_C( 25074), -INT16_C(  7260) },
       INT32_C(          61),
      { -INT16_C( 22943),  INT16_C(     0), -INT16_C( 12517),  INT16_C(     0), -INT16_C( 18283), -INT16_C( 31264),  INT16_C( 21441),  INT16_C(     0) } },
    { UINT8_C( 49),
      { -INT16_C( 27197),  INT16_C( 27751),  INT16_C( 12361), -INT16_C( 25329), -INT16_C( 16033), -INT16_C( 27992),  INT16_C( 14057),  INT16_C( 24944) },
      {  INT16_C(  2315), -INT16_C( 12533),  INT16_C( 14863), -INT16_C( 21027), -INT16_C( 15905),  INT16_C( 13099),  INT16_C( 28206), -INT16_C(  3483) },
       INT32_C(           3),
      { -INT16_C( 20968),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(  2814), -INT16_C( 27327),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C(252),
      { -INT16_C(  5523),  INT16_C( 12123), -INT16_C(  4462), -INT16_C( 14312),  INT16_C( 31326),  INT16_C( 26836), -INT16_C( 23675), -INT16_C( 16521) },
      {  INT16_C(  9345),  INT16_C( 17054), -INT16_C( 11697), -INT16_C( 16784),  INT16_C( 25143),  INT16_C(   961),  INT16_C(  3777),  INT16_C( 12031) },
       INT32_C(          56),
      {  INT16_C(     0),  INT16_C(     0), -INT16_C( 27950),  INT16_C(  6334),  INT16_C( 24162), -INT16_C( 11261), -INT16_C( 31474),  INT16_C( 30510) } },
    { UINT8_C( 73),
      {  INT16_C( 21366), -INT16_C(  3929),  INT16_C(  3879), -INT16_C( 13707),  INT16_C( 13446), -INT16_C( 21685), -INT16_C( 29229), -INT16_C( 23046) },
      { -INT16_C( 18178),  INT16_C( 24796), -INT16_C(  8326), -INT16_C( 30687),  INT16_C( 20702),  INT16_C( 14720),  INT16_C(  2733),  INT16_C(  9090) },
       INT32_C(          29),
      { -INT16_C( 10465),  INT16_C(     0),  INT16_C(     0), -INT16_C( 20220),  INT16_C(     0),  INT16_C(     0),  INT16_C( 24917),  INT16_C(     0) } },
    { UINT8_C( 57),
      {  INT16_C( 20360), -INT16_C( 16960),  INT16_C( 27546),  INT16_C( 10384),  INT16_C( 13669),  INT16_C(  7718), -INT16_C( 31215), -INT16_C(  3944) },
      {  INT16_C(  8360), -INT16_C(  1842),  INT16_C(  2208), -INT16_C( 21851), -INT16_C( 13942), -INT16_C( 19448), -INT16_C( 29476),  INT16_C( 26094) },
       INT32_C(          27),
      {  INT16_C( 16645),  INT16_C(     0),  INT16_C(     0), -INT16_C( 31403),  INT16_C( 11852),  INT16_C( 13728),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C( 25),
      { -INT16_C( 24910), -INT16_C(  6274), -INT16_C( 25404),  INT16_C( 19192), -INT16_C(  6092),  INT16_C( 21746), -INT16_C(  5450), -INT16_C( 16652) },
      { -INT16_C( 24688),  INT16_C( 22857), -INT16_C(   601),  INT16_C( 13109), -INT16_C( 25877), -INT16_C( 26353), -INT16_C( 31300),  INT16_C( 28338) },
       INT32_C(          35),
      { -INT16_C(  2668),  INT16_C(     0),  INT16_C(     0),  INT16_C( 22465),  INT16_C( 16804),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT8_C(205),
      {  INT16_C( 12621),  INT16_C( 13570),  INT16_C( 22052),  INT16_C(  3820), -INT16_C( 21941), -INT16_C(  5474), -INT16_C(  2061), -INT16_C(  3695) },
      { -INT16_C( 15315), -INT16_C( 14372),  INT16_C( 30419),  INT16_C( 22660), -INT16_C(  3544),  INT16_C( 22907),  INT16_C( 25160), -INT16_C( 27353) },
       INT32_C(          20),
      {  INT16_C(  5340),  INT16_C(     0),  INT16_C( 25159), -INT16_C(  4411),  INT16_C(     0),  INT16_C(     0),  INT16_C( 32566),  INT16_C(  6425) } },
    { UINT8_C(127),
      { -INT16_C( 14665),  INT16_C( 25034), -INT16_C( 19355),  INT16_C( 23637),  INT16_C( 17989),  INT16_C(  2697),  INT16_C( 20770), -INT16_C( 26403) },
      {  INT16_C( 14037), -INT16_C( 14399),  INT16_C(  6833),  INT16_C(  5135), -INT16_C( 23231),  INT16_C( 27304),  INT16_C( 24688),  INT16_C( 10218) },
       INT32_C(          38),
      { -INT16_C( 21043),  INT16_C( 29361),  INT16_C(  6470),  INT16_C(  5445), -INT16_C( 28311), -INT16_C( 23974),  INT16_C( 18584),  INT16_C(     0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    int       imm8 = test_vec[i].imm8;
    easysimd__m128i r = easysimd_mm_loadu_si128(test_vec[i].r);
    easysimd__m128i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      EASYSIMD_CONSTIFY_64_(easysimd_mm_maskz_shldi_epi16, ret, easysimd_mm_setzero_si128(), imm8, k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_maskz_shldi_epi16");
    easysimd_assert_m128i_i16(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i16x8();
    easysimd__m128i b = easysimd_test_x86_random_i16x8();
    int       imm8 = easysimd_test_codegen_random_i32() & 63;
    easysimd__m128i r;
    EASYSIMD_CONSTIFY_64_(easysimd_mm_maskz_shldi_epi16, r, easysimd_mm_setzero_si128(), imm8, k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_shldi_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    int32_t    a[4];
    int32_t    b[4];
    int        imm8;
    int32_t    r[4];
  } test_vec[8] = {
    { UINT8_C( 29),
      {  INT32_C(   865172593),  INT32_C(  1850141840), -INT32_C(   815135428), -INT32_C(    27171122) },
      { -INT32_C(    11080378),  INT32_C(   895313225), -INT32_C(  1740702351), -INT32_C(  1632275923) },
       INT32_C(          47),
      { -INT32_C(  1103560789),  INT32_C(           0),  INT32_C(    43928607), -INT32_C(  1285075110) } },
    { UINT8_C( 51),
      {  INT32_C(   493939992), -INT32_C(  1108656297), -INT32_C(  1929123424), -INT32_C(  1529478337) },
      {  INT32_C(  1561660256), -INT32_C(  1131762358), -INT32_C(  1431623581),  INT32_C(   316484346) },
       INT32_C(          23),
      { -INT32_C(  1943106939), -INT32_C(  1411496618),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(140),
      {  INT32_C(    53226522),  INT32_C(   860010800), -INT32_C(  1684806000), -INT32_C(  1461325573) },
      { -INT32_C(  1559518597),  INT32_C(  2023601485), -INT32_C(   527454317), -INT32_C(   127009058) },
       INT32_C(          43),
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(  1623947516),  INT32_C(   797433795) } },
    { UINT8_C( 83),
      {  INT32_C(   602115645),  INT32_C(   287211041),  INT32_C(   109889380), -INT32_C(  2007813934) },
      {  INT32_C(  2098973645),  INT32_C(  1499200347),  INT32_C(    42226792), -INT32_C(  1923752113) },
       INT32_C(          45),
      {  INT32_C(  1908912035), -INT32_C(   809227477),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(182),
      { -INT32_C(  1776639793),  INT32_C(  1550327852),  INT32_C(  1059778676),  INT32_C(   127575819) },
      {  INT32_C(  1450177538),  INT32_C(  1487303032), -INT32_C(    20630586), -INT32_C(  1296739101) },
       INT32_C(          52),
      {  INT32_C(           0),  INT32_C(    46500455),  INT32_C(   122678355),  INT32_C(           0) } },
    { UINT8_C(240),
      { -INT32_C(  1587249743),  INT32_C(  1320002471),  INT32_C(   844149822), -INT32_C(  1783978205) },
      { -INT32_C(  2141519283),  INT32_C(   476273352), -INT32_C(   569371121),  INT32_C(   265171038) },
       INT32_C(          45),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(214),
      {  INT32_C(   303342174), -INT32_C(  1707784461), -INT32_C(   202913040), -INT32_C(  2135202010) },
      { -INT32_C(   527444022),  INT32_C(   390032871),  INT32_C(  1858424380),  INT32_C(  1564744191) },
       INT32_C(          60),
      {  INT32_C(           0),  INT32_C(   829683422),  INT32_C(   116151523),  INT32_C(           0) } },
    { UINT8_C(158),
      {  INT32_C(  1871661477), -INT32_C(  1751809488),  INT32_C(   341906749),  INT32_C(   318456484) },
      { -INT32_C(   816967039), -INT32_C(   238043945),  INT32_C(  1504521217),  INT32_C(  1106795676) },
       INT32_C(          37),
      {  INT32_C(           0), -INT32_C(   223328738), -INT32_C(  1943885909),  INT32_C(  1600672904) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    int       imm8 = test_vec[i].imm8;
    easysimd__m128i r = easysimd_mm_loadu_si128(test_vec[i].r);
    easysimd__m128i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      EASYSIMD_CONSTIFY_64_(easysimd_mm_maskz_shldi_epi32, ret, easysimd_mm_setzero_si128(), imm8, k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_maskz_shldi_epi32");
    easysimd_assert_m128i_i32(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    int       imm8 = easysimd_test_codegen_random_i32() & 63;
    easysimd__m128i r;
    EASYSIMD_CONSTIFY_64_(easysimd_mm_maskz_shldi_epi32, r, easysimd_mm_setzero_si128(), imm8, k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_shldi_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    int64_t    a[2];
    int64_t    b[2];
    int        imm8;
    int64_t    r[2];
  } test_vec[8] = {
    { UINT8_C(231),
      {  INT64_C(  697402542533110472), -INT64_C( 7193987114297899762) },
      { -INT64_C( 3898989001360555311),  INT64_C( 9153229474769297722) },
       INT32_C(          55),
      {  INT64_C( 7234172987918475233), -INT64_C( 8701091477271371472) } },
    { UINT8_C(212),
      {  INT64_C(  370596112083626505), -INT64_C( 9100634105567592248) },
      { -INT64_C( 8283849519997944335), -INT64_C( 1234015005288658719) },
       INT32_C(          15),
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 97),
      { -INT64_C( 6007603307363055519), -INT64_C( 5213496157780762264) },
      {  INT64_C( 6586412709445120762), -INT64_C(  890394883084561022) },
       INT32_C(          54),
      {  INT64_C( 1753828699081319948),  INT64_C(                   0) } },
    { UINT8_C(130),
      {  INT64_C( 5742880485368578484), -INT64_C( 6548572496381515147) },
      {  INT64_C( 1304473121602369843), -INT64_C( 3071562552555533041) },
       INT32_C(          32),
      {  INT64_C(                   0), -INT64_C( 6392461141752177119) } },
    { UINT8_C( 25),
      {  INT64_C( 2650708511105617501),  INT64_C(   86095706679897939) },
      {  INT64_C( 7721402650487506169),  INT64_C(  265864826130791502) },
       INT32_C(          25),
      { -INT64_C( 5652268097607413795),  INT64_C(                   0) } },
    { UINT8_C(116),
      {  INT64_C( 5447548415922083220), -INT64_C( 7289275369108759610) },
      { -INT64_C( 7488500955925167503),  INT64_C( 8494671673779361584) },
       INT32_C(          55),
      {  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 63),
      {  INT64_C( 8960209643955940882),  INT64_C( 6644598201857733606) },
      {  INT64_C( 2551002218611134318),  INT64_C( 7406449226245922014) },
       INT32_C(          44),
      {  INT64_C( 6836783326522509400), -INT64_C( 2360336538238891883) } },
    { UINT8_C(220),
      { -INT64_C( 2072523852742485140),  INT64_C( 2373632457763297313) },
      {  INT64_C( 7813049835509912381),  INT64_C( 6122775527906923599) },
       INT32_C(          19),
      {  INT64_C(                   0),  INT64_C(                   0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m128i a = easysimd_mm_loadu_si128(test_vec[i].a);
    easysimd__m128i b = easysimd_mm_loadu_si128(test_vec[i].b);
    int       imm8 = test_vec[i].imm8;
    easysimd__m128i r = easysimd_mm_loadu_si128(test_vec[i].r);
    easysimd__m128i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      EASYSIMD_CONSTIFY_64_(easysimd_mm_maskz_shldi_epi64, ret, easysimd_mm_setzero_si128(), imm8, k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm_maskz_shldi_epi64");
    easysimd_assert_m128i_i64(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i64x2();
    easysimd__m128i b = easysimd_test_x86_random_i64x2();
    int       imm8 = easysimd_test_codegen_random_i32() & 63;
    easysimd__m128i r;
    EASYSIMD_CONSTIFY_64_(easysimd_mm_maskz_shldi_epi64, r, easysimd_mm_setzero_si128(), imm8, k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x2(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x2(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_shldi_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask16 k;
    int16_t    a[16];
    int16_t    b[16];
    int        imm8;
    int16_t    r[16];
  } test_vec[8] = {
    { UINT16_C(57425),
      { -INT16_C( 18607),  INT16_C(  5328), -INT16_C( 24909),  INT16_C( 16413),  INT16_C( 20232), -INT16_C( 15081), -INT16_C( 32411),  INT16_C( 23183),
         INT16_C( 26385),  INT16_C(  2155), -INT16_C( 29861),  INT16_C(  9946),  INT16_C( 10500),  INT16_C( 21474),  INT16_C( 13129), -INT16_C( 26061) },
      {  INT16_C(  1258), -INT16_C( 25170), -INT16_C( 13406), -INT16_C( 21539), -INT16_C(  2789), -INT16_C( 32656), -INT16_C(   138), -INT16_C( 30502),
         INT16_C( 18022), -INT16_C( 15984),  INT16_C( 27345), -INT16_C( 10777), -INT16_C( 13933), -INT16_C(  9176),  INT16_C( 23804), -INT16_C(  6538) },
       INT32_C(          32),
      { -INT16_C( 18607),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 20232),  INT16_C(     0), -INT16_C( 32411),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 21474),  INT16_C( 13129), -INT16_C( 26061) } },
    { UINT16_C(24816),
      {  INT16_C(  2989),  INT16_C(  7509), -INT16_C( 13172),  INT16_C( 26140), -INT16_C( 32172), -INT16_C(  6996),  INT16_C( 32067),  INT16_C( 10830),
        -INT16_C(  7854),  INT16_C( 31731), -INT16_C(  4163),  INT16_C( 13271),  INT16_C( 14293),  INT16_C( 22616),  INT16_C( 18745), -INT16_C(  6215) },
      {  INT16_C(  3668), -INT16_C(  8188),  INT16_C(  8666),  INT16_C( 11847), -INT16_C(  3165), -INT16_C(  6382),  INT16_C( 24689), -INT16_C( 15599),
         INT16_C(  1345), -INT16_C(   450),  INT16_C(  5620), -INT16_C( 13774), -INT16_C( 30132), -INT16_C( 31198), -INT16_C(  9261),  INT16_C( 10349) },
       INT32_C(          42),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 21454), -INT16_C( 19556),  INT16_C(  3457),  INT16_C( 15116),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 25112), -INT16_C(  6289),  INT16_C(     0) } },
    { UINT16_C(20370),
      {  INT16_C( 14067),  INT16_C(  1347), -INT16_C( 19427),  INT16_C( 11878), -INT16_C( 22665), -INT16_C( 18893),  INT16_C( 10406), -INT16_C( 10037),
         INT16_C(  6386),  INT16_C(  5218),  INT16_C( 13982),  INT16_C(  3056), -INT16_C(  9634),  INT16_C( 26236),  INT16_C(  3998), -INT16_C( 28234) },
      { -INT16_C(  1723),  INT16_C( 25239), -INT16_C(   595),  INT16_C(  9360), -INT16_C( 15196),  INT16_C( 19162), -INT16_C( 22804), -INT16_C(  8670),
        -INT16_C( 31298),  INT16_C( 23794), -INT16_C(  7493),  INT16_C(  6503), -INT16_C(  7236),  INT16_C( 23423),  INT16_C( 13810),  INT16_C( 14316) },
       INT32_C(          46),
      {  INT16_C(     0), -INT16_C( 10075),  INT16_C(     0),  INT16_C(     0), -INT16_C(  3799),  INT16_C(     0),  INT16_C(     0), -INT16_C(  2168),
        -INT16_C( 24209), -INT16_C( 26820), -INT16_C( 18258),  INT16_C(  1625),  INT16_C(     0),  INT16_C(     0), -INT16_C( 29316),  INT16_C(     0) } },
    { UINT16_C(10880),
      {  INT16_C(  9472), -INT16_C(  9490), -INT16_C(  9617), -INT16_C( 28032),  INT16_C( 16056), -INT16_C( 21993), -INT16_C( 11622),  INT16_C(   397),
         INT16_C( 18923),  INT16_C( 27365), -INT16_C( 10332), -INT16_C( 28256), -INT16_C( 12785), -INT16_C( 22508), -INT16_C( 27222), -INT16_C( 21806) },
      { -INT16_C( 16198),  INT16_C( 10628),  INT16_C(  1434),  INT16_C( 21179), -INT16_C( 11709), -INT16_C(  8451), -INT16_C( 30044), -INT16_C( 28705),
        -INT16_C( 15149),  INT16_C( 30970), -INT16_C( 25956), -INT16_C( 21751),  INT16_C(  7528),  INT16_C(  4691),  INT16_C(  9906),  INT16_C( 27836) },
       INT32_C(          38),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 25443),
         INT16_C(     0), -INT16_C( 18082),  INT16_C(     0),  INT16_C( 26666),  INT16_C(     0),  INT16_C(  1284),  INT16_C(     0),  INT16_C(     0) } },
    { UINT16_C(20806),
      { -INT16_C( 30253), -INT16_C( 12252), -INT16_C( 14233),  INT16_C( 18266),  INT16_C( 11864),  INT16_C( 21003), -INT16_C( 22618), -INT16_C( 20500),
         INT16_C( 21586), -INT16_C( 22836),  INT16_C( 32615),  INT16_C(  9164), -INT16_C( 19733), -INT16_C( 32412), -INT16_C( 21965),  INT16_C(  2003) },
      { -INT16_C(  2252), -INT16_C( 25641),  INT16_C( 12991),  INT16_C(  6114), -INT16_C(  4512),  INT16_C(  1641),  INT16_C( 21909), -INT16_C(  5963),
        -INT16_C( 32342),  INT16_C(  4494),  INT16_C( 23040), -INT16_C(  5068), -INT16_C( 26356),  INT16_C( 16493),  INT16_C( 16451),  INT16_C( 30535) },
       INT32_C(          55),
      {  INT16_C(     0),  INT16_C(  4685),  INT16_C( 13209),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 11478),  INT16_C(     0),
         INT16_C( 10560),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 30156),  INT16_C(     0),  INT16_C(  6560),  INT16_C(     0) } },
    { UINT16_C(62800),
      { -INT16_C( 20466),  INT16_C( 30947),  INT16_C( 31158),  INT16_C( 27597),  INT16_C( 30561), -INT16_C(  4115), -INT16_C(  4728), -INT16_C( 17079),
         INT16_C( 21977),  INT16_C( 18262), -INT16_C( 26219), -INT16_C(  9081), -INT16_C( 16623),  INT16_C(  9467),  INT16_C( 19382), -INT16_C( 15335) },
      { -INT16_C(   516), -INT16_C( 19908),  INT16_C(  2678), -INT16_C( 10466),  INT16_C(  2945),  INT16_C(  2758),  INT16_C(  4088), -INT16_C( 11577),
         INT16_C(  7524), -INT16_C(  1511), -INT16_C( 24394), -INT16_C( 14378), -INT16_C( 11937),  INT16_C(  5611),  INT16_C(  1309),  INT16_C(  6618) },
       INT32_C(           2),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  8828),  INT16_C(     0), -INT16_C( 18912),  INT16_C(     0),
         INT16_C( 22372),  INT16_C(     0),  INT16_C( 26198),  INT16_C(     0), -INT16_C(   953), -INT16_C( 27668),  INT16_C( 11992),  INT16_C(  4196) } },
    { UINT16_C(59680),
      { -INT16_C( 23985),  INT16_C(  5620), -INT16_C(  4692),  INT16_C( 29476), -INT16_C( 30529), -INT16_C( 10096),  INT16_C( 18050),  INT16_C( 22904),
        -INT16_C( 10226), -INT16_C(  1750),  INT16_C( 18413), -INT16_C( 14338),  INT16_C(    96),  INT16_C( 11486), -INT16_C(   392), -INT16_C( 14571) },
      {  INT16_C(  2720),  INT16_C( 19676),  INT16_C(   247), -INT16_C( 18753),  INT16_C( 20361),  INT16_C(  2958),  INT16_C(  1686), -INT16_C( 23452),
        -INT16_C( 28706), -INT16_C( 13155), -INT16_C( 25386),  INT16_C( 14227),  INT16_C( 29084),  INT16_C(  5475),  INT16_C( 30832),  INT16_C(  4316) },
       INT32_C(           2),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 25152),  INT16_C(     0),  INT16_C(     0),
         INT16_C( 24634),  INT16_C(     0),  INT16_C(     0),  INT16_C(  8184),  INT16_C(     0), -INT16_C( 19592), -INT16_C(  1567),  INT16_C(  7252) } },
    { UINT16_C( 7353),
      {  INT16_C( 16943), -INT16_C( 17044),  INT16_C(   590), -INT16_C( 19772), -INT16_C( 23898),  INT16_C( 17217),  INT16_C(  6254),  INT16_C(   735),
         INT16_C( 31823), -INT16_C( 19853), -INT16_C(  7279),  INT16_C( 27946), -INT16_C( 21004),  INT16_C( 20774), -INT16_C(  8154),  INT16_C( 22125) },
      { -INT16_C(  9950),  INT16_C( 28691), -INT16_C( 10277), -INT16_C( 32477),  INT16_C( 25722), -INT16_C(  5947), -INT16_C( 23428), -INT16_C( 13334),
         INT16_C( 24096), -INT16_C( 20099), -INT16_C( 22463),  INT16_C( 13599),  INT16_C( 17749),  INT16_C( 31622), -INT16_C(  3035),  INT16_C( 18641) },
       INT32_C(          13),
      { -INT16_C(  1244),  INT16_C(     0),  INT16_C(     0), -INT16_C( 28636), -INT16_C( 13169),  INT16_C( 15640),  INT16_C(     0), -INT16_C(  1667),
         INT16_C(     0),  INT16_C(     0),  INT16_C( 13576),  INT16_C( 18083), -INT16_C( 30550),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    int       imm8 = test_vec[i].imm8;
    easysimd__m256i r = easysimd_mm256_loadu_si256(test_vec[i].r);
    easysimd__m256i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      EASYSIMD_CONSTIFY_64_(easysimd_mm256_maskz_shldi_epi16, ret, easysimd_mm256_setzero_si256(), imm8, k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_maskz_shldi_epi16");
    easysimd_assert_m256i_i16(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m256i a = easysimd_test_x86_random_i16x16();
    easysimd__m256i b = easysimd_test_x86_random_i16x16();
    int       imm8 = easysimd_test_codegen_random_i32() & 63;
    easysimd__m256i r;
    EASYSIMD_CONSTIFY_64_(easysimd_mm256_maskz_shldi_epi16, r, easysimd_mm256_setzero_si256(), imm8, k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_shldi_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    int32_t    a[8];
    int32_t    b[8];
    int        imm8;
    int32_t    r[8];
  } test_vec[8] = {
    { UINT8_C(197),
      {  INT32_C(   892245276),  INT32_C(  1491484883), -INT32_C(   761409118),  INT32_C(  1484629115), -INT32_C(   352691393),  INT32_C(   971628660), -INT32_C(   776007402),  INT32_C(   580277766) },
      { -INT32_C(  1101478677), -INT32_C(  1508426236), -INT32_C(  1955023857), -INT32_C(  1746668200), -INT32_C(   981344944), -INT32_C(   721524034), -INT32_C(  2103067268), -INT32_C(    22725869) },
       INT32_C(          63),
      {  INT32_C(  1596744309),  INT32_C(           0),  INT32_C(  1169971719),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1095950014),  INT32_C(  2136120713) } },
    { UINT8_C( 59),
      { -INT32_C(  2008372780),  INT32_C(   400610594), -INT32_C(  1771538503),  INT32_C(  1666461176), -INT32_C(   388028373),  INT32_C(   150692301),  INT32_C(    50854150), -INT32_C(  1975645514) },
      { -INT32_C(   703428172),  INT32_C(   401470046),  INT32_C(  1655592297), -INT32_C(  1379597694), -INT32_C(   124410837),  INT32_C(   201428997),  INT32_C(  1074727050), -INT32_C(   942977517) },
       INT32_C(          23),
      { -INT32_C(   362084028), -INT32_C(  1861486855),  INT32_C(           0), -INT32_C(    61414783),  INT32_C(   368855762), -INT32_C(   427425592),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(208),
      { -INT32_C(   533115510), -INT32_C(    43869189),  INT32_C(    69734496),  INT32_C(   906633637),  INT32_C(   717231650),  INT32_C(  1933377573), -INT32_C(  1454766901),  INT32_C(   746160289) },
      { -INT32_C(   938691891), -INT32_C(  1362792882),  INT32_C(   615771774),  INT32_C(   828030223), -INT32_C(   128247085), -INT32_C(   429090534),  INT32_C(  1066383005),  INT32_C(    57346102) },
       INT32_C(          58),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1948160917),  INT32_C(           0),  INT32_C(   754859738), -INT32_C(  2079478752) } },
    { UINT8_C(229),
      {  INT32_C(  2137307025),  INT32_C(   663652458), -INT32_C(    50675742), -INT32_C(  1290341862),  INT32_C(   357694558), -INT32_C(  1823764341),  INT32_C(  1917734907), -INT32_C(  1420339686) },
      {  INT32_C(  1999289101),  INT32_C(   631158851), -INT32_C(  1843292041), -INT32_C(   381339509), -INT32_C(  1057057228),  INT32_C(   559106598), -INT32_C(  1282170471),  INT32_C(    90172408) },
       INT32_C(          38),
      { -INT32_C(   651303843),  INT32_C(           0),  INT32_C(  1051719844),  INT32_C(           0),  INT32_C(           0), -INT32_C(   756800824), -INT32_C(  1819017492), -INT32_C(   707426687) } },
    { UINT8_C( 66),
      { -INT32_C(  1262940389),  INT32_C(  1765755697),  INT32_C(   664610961),  INT32_C(  1900961063),  INT32_C(  1376415665), -INT32_C(   297091837), -INT32_C(  1500229604), -INT32_C(   387416371) },
      { -INT32_C(  1080254066),  INT32_C(  2116606957),  INT32_C(   732284164), -INT32_C(   727911645),  INT32_C(  1713808995), -INT32_C(  2141949597), -INT32_C(  1910052671), -INT32_C(   176812441) },
       INT32_C(          47),
      {  INT32_C(           0), -INT32_C(  1516716268),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   672024339),  INT32_C(           0) } },
    { UINT8_C(238),
      { -INT32_C(  1577968932), -INT32_C(  1278992960),  INT32_C(  1595316409),  INT32_C(   818117823), -INT32_C(  1175370800),  INT32_C(  1998618473),  INT32_C(   136713717), -INT32_C(  1510489143) },
      { -INT32_C(  1656296995), -INT32_C(  1068430585),  INT32_C(  1646290851), -INT32_C(  1265441820), -INT32_C(  1888648410), -INT32_C(   133788414),  INT32_C(  1795173538), -INT32_C(   854525969) },
       INT32_C(          33),
      {  INT32_C(           0),  INT32_C(  1736981377), -INT32_C(  1104334478),  INT32_C(  1636235647),  INT32_C(           0), -INT32_C(   297730349),  INT32_C(   273427434),  INT32_C(  1273989011) } },
    { UINT8_C( 98),
      {  INT32_C(   587573435), -INT32_C(  1425577784),  INT32_C(  2127674362), -INT32_C(  1233100759), -INT32_C(  1822853018),  INT32_C(  1887618169),  INT32_C(   743526357),  INT32_C(  1972255162) },
      { -INT32_C(  1432841246), -INT32_C(   162095109), -INT32_C(  2072762533), -INT32_C(   298126200), -INT32_C(   427715731),  INT32_C(   760611928),  INT32_C(   223979603), -INT32_C(  1014765599) },
       INT32_C(          58),
      {  INT32_C(           0),  INT32_C(   601447039),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   457877487),  INT32_C(  1412785825),  INT32_C(           0) } },
    { UINT8_C(187),
      { -INT32_C(   350851900), -INT32_C(   730621216),  INT32_C(  1749115604), -INT32_C(   406837277),  INT32_C(   641461630),  INT32_C(   755451974), -INT32_C(   408433717), -INT32_C(    39707335) },
      {  INT32_C(  1776859273),  INT32_C(   641621074), -INT32_C(  1584496706),  INT32_C(   629755559), -INT32_C(  2125740997), -INT32_C(   659664372),  INT32_C(  1455379997), -INT32_C(    44867213) },
       INT32_C(          25),
      { -INT32_C(  1999384207), -INT32_C(  1068729160),  INT32_C(           0), -INT32_C(   968158563), -INT32_C(    50161784), -INT32_C(  1917756252),  INT32_C(           0),  INT32_C(  1945806530) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    int       imm8 = test_vec[i].imm8;
    easysimd__m256i r = easysimd_mm256_loadu_si256(test_vec[i].r);
    easysimd__m256i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      EASYSIMD_CONSTIFY_64_(easysimd_mm256_maskz_shldi_epi32, ret, easysimd_mm256_setzero_si256(), imm8, k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_maskz_shldi_epi32");
    easysimd_assert_m256i_i32(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    int       imm8 = easysimd_test_codegen_random_i32() & 63;
    easysimd__m256i r;
    EASYSIMD_CONSTIFY_64_(easysimd_mm256_maskz_shldi_epi32, r, easysimd_mm256_setzero_si256(), imm8, k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_shldi_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    int64_t    a[4];
    int64_t    b[4];
    int        imm8;
    int64_t    r[4];
  } test_vec[8] = {  
    { UINT8_C(239),
      { -INT64_C( 4037548708761623636), -INT64_C( 3630197366639193862), -INT64_C( 8688133402930899945),  INT64_C( 2237153713695947892) },
      { -INT64_C( 5640352023443736550), -INT64_C( 6086377862594977279), -INT64_C( 6690733663466773781),  INT64_C(   21616287700479929) },
       INT32_C(           3),
      {  INT64_C( 4593098477326114149),  INT64_C( 7851909214305552341),  INT64_C( 4281909071391006909), -INT64_C(  549514364141968480) } },
    { UINT8_C( 55),
      {  INT64_C( 4315280104620633003), -INT64_C( 7183869372301668739),  INT64_C( 8916708008295526633),  INT64_C( 6759262788117249999) },
      {  INT64_C( 8612164493556516580),  INT64_C( 6455598285478865691),  INT64_C( 4467553712829754324), -INT64_C( 2612756323974889809) },
       INT32_C(          37),
      { -INT64_C( 4096739166680662199), -INT64_C( 1958556028729782785),  INT64_C(  774088242516331984),  INT64_C(                   0) } },
    { UINT8_C(246),
      { -INT64_C( 7871434880350873687),  INT64_C( 4538479772118162593), -INT64_C( 6188142656952125206), -INT64_C( 4005407475211699445) },
      {  INT64_C( 2902535354032356103), -INT64_C( 8968724489773404676), -INT64_C( 3558745563079150104), -INT64_C( 2112562214257693498) },
       INT32_C(          38),
      {  INT64_C(                   0),  INT64_C( 6451169387808040466),  INT64_C( 4273417639699869948),  INT64_C(                   0) } },
    { UINT8_C(170),
      {  INT64_C( 4117681532396098782),  INT64_C( 2211852717719136122),  INT64_C( 7888276585032482992), -INT64_C( 3392266109585237385) },
      { -INT64_C(   11398063174413548), -INT64_C( 4784616054113503192), -INT64_C( 4733053101679215327),  INT64_C( 4030419451626185516) },
       INT32_C(           0),
      {  INT64_C(                   0),  INT64_C( 2211852717719136122),  INT64_C(                   0), -INT64_C( 3392266109585237385) } },
    { UINT8_C(204),
      { -INT64_C( 7732556368448322389), -INT64_C( 7435836852487754487),  INT64_C( 8858829565708242153), -INT64_C( 6347822596255766575) },
      {  INT64_C( 6236354417009613828), -INT64_C( 4339029287360446103),  INT64_C( 4629278684510981431),  INT64_C( 4416417398934747559) },
       INT32_C(          33),
      {  INT64_C(                   0),  INT64_C(                   0), -INT64_C( 7239226534131859233),  INT64_C( 4176050313739143283) } },
    { UINT8_C(144),
      {  INT64_C( 5466008339948880832), -INT64_C( 5697538209248731150), -INT64_C( 3388683241478738042),  INT64_C( 3128416704178238788) },
      {  INT64_C( 9155264497431896866),  INT64_C( 6811391902508552615),  INT64_C( 2123189769589349915), -INT64_C( 1666225455554467765) },
       INT32_C(           4),
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 21),
      { -INT64_C( 5034300018731986759), -INT64_C( 5417973704987082471),  INT64_C( 4837627838360186100),  INT64_C( 8050000400373713813) },
      {  INT64_C( 2211789710676292392), -INT64_C(  526950821823936676), -INT64_C(  578582385812787503), -INT64_C( 9176558850900623487) },
       INT32_C(           3),
      { -INT64_C( 3380912002436790840),  INT64_C(                   0),  INT64_C( 1807534559462385575),  INT64_C(                   0) } },
    { UINT8_C( 59),
      { -INT64_C( 2111182182036281560),  INT64_C( 8453934898975320897),  INT64_C( 3726498770774870368), -INT64_C( 2922905774399735595) },
      { -INT64_C( 2628350000471079001),  INT64_C(  168916471958354660),  INT64_C( 6015711232447827814), -INT64_C( 2605342060986272943) },
       INT32_C(          62),
      {  INT64_C( 3954598518309618153),  INT64_C( 4653915136416976569),  INT64_C(                   0),  INT64_C( 8572036521608207572) } }
 };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_mm256_loadu_si256(test_vec[i].a);
    easysimd__m256i b = easysimd_mm256_loadu_si256(test_vec[i].b);
    int       imm8 = test_vec[i].imm8;
    easysimd__m256i r = easysimd_mm256_loadu_si256(test_vec[i].r);
    easysimd__m256i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      EASYSIMD_CONSTIFY_64_(easysimd_mm256_maskz_shldi_epi64, ret, easysimd_mm256_setzero_si256(), imm8, k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm256_maskz_shldi_epi64");
    easysimd_assert_m256i_i64(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i64x4();
    easysimd__m256i b = easysimd_test_x86_random_i64x4();
    int       imm8 = easysimd_test_codegen_random_i32() & 63;
    easysimd__m256i r;
    EASYSIMD_CONSTIFY_64_(easysimd_mm256_maskz_shldi_epi64, r, easysimd_mm256_setzero_si256(), imm8, k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_shldi_epi16(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask32 k;
    int16_t    a[32];
    int16_t    b[32];
    int        imm8;
    int16_t    r[32];
  } test_vec[8] = {
    { UINT32_C( 121873807),
      {  INT16_C( 20140),  INT16_C( 29748), -INT16_C( 32029),  INT16_C( 16943),  INT16_C(  5124), -INT16_C( 19255), -INT16_C(  1627),  INT16_C( 14708),
         INT16_C( 27875),  INT16_C(   947),  INT16_C( 12554),  INT16_C( 18721), -INT16_C( 18145), -INT16_C( 20808), -INT16_C(   929),  INT16_C(  2997),
        -INT16_C(  5814),  INT16_C( 11903), -INT16_C( 20629),  INT16_C( 28528),  INT16_C( 14787),  INT16_C( 26660), -INT16_C( 26574),  INT16_C(  5538),
         INT16_C( 21764),  INT16_C(  3608),  INT16_C( 14726), -INT16_C( 23209),  INT16_C(  4339),  INT16_C( 21075),  INT16_C(  2316),  INT16_C( 22109) },
      { -INT16_C(  8718),  INT16_C( 24196), -INT16_C(  2932),  INT16_C( 20429), -INT16_C(  3795),  INT16_C( 24503),  INT16_C( 22921), -INT16_C( 29068),
        -INT16_C( 29265),  INT16_C( 13724), -INT16_C(  2874), -INT16_C( 17957),  INT16_C( 11780),  INT16_C(  4107),  INT16_C( 26935),  INT16_C( 10854),
        -INT16_C(  5306), -INT16_C( 11640),  INT16_C( 21983),  INT16_C(  3361), -INT16_C( 10169), -INT16_C( 12180), -INT16_C(  7886), -INT16_C(  7842),
        -INT16_C(  1170),  INT16_C( 13334), -INT16_C(  3601), -INT16_C(  3090), -INT16_C(  1760),  INT16_C( 22275),  INT16_C( 26978), -INT16_C( 22399) },
       INT32_C(          20),
      { -INT16_C(  5427),  INT16_C( 17221),  INT16_C( 11839),  INT16_C(  8948),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 26808),
        -INT16_C( 12744),  INT16_C(     0),  INT16_C(  4271),  INT16_C(     0),  INT16_C(     0), -INT16_C(  5247),  INT16_C(     0), -INT16_C( 17582),
        -INT16_C( 27474), -INT16_C(  6147),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 31954),  INT16_C(     0),
         INT16_C( 20559), -INT16_C(  7805), -INT16_C( 26513),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0) } },
    { UINT32_C(2789317471),
      { -INT16_C( 21132), -INT16_C( 22922), -INT16_C( 10866), -INT16_C(   889), -INT16_C( 25136), -INT16_C( 16591),  INT16_C(  8079), -INT16_C( 20558),
        -INT16_C( 19176),  INT16_C( 31494), -INT16_C( 30690),  INT16_C( 29475), -INT16_C( 24943), -INT16_C(  3929), -INT16_C(  6087), -INT16_C( 21098),
         INT16_C(  3477),  INT16_C(  9299), -INT16_C(  9502), -INT16_C( 19936),  INT16_C( 20856),  INT16_C(  1905),  INT16_C(  9072), -INT16_C( 30282),
        -INT16_C( 17192), -INT16_C(  2556),  INT16_C( 10052), -INT16_C( 10647),  INT16_C(  4293), -INT16_C(    58),  INT16_C( 24056), -INT16_C( 29012) },
      {  INT16_C(   106),  INT16_C( 19634), -INT16_C( 11558),  INT16_C( 21246),  INT16_C( 28452), -INT16_C( 27559),  INT16_C(  3986),  INT16_C( 27165),
         INT16_C(  8652),  INT16_C(  4192), -INT16_C( 13751),  INT16_C(  3814), -INT16_C( 21030), -INT16_C( 11507), -INT16_C( 17910),  INT16_C( 29793),
         INT16_C(  5050), -INT16_C( 27456), -INT16_C( 16667),  INT16_C(  2535),  INT16_C( 16429), -INT16_C( 16482), -INT16_C( 17584),  INT16_C(  7209),
        -INT16_C( 30243),  INT16_C(  9772),  INT16_C(  4947),  INT16_C( 11828),  INT16_C( 17088), -INT16_C( 13823),  INT16_C( 25340), -INT16_C( 18882) },
       INT32_C(          53),
      { -INT16_C( 20864), -INT16_C( 12599), -INT16_C( 20006), -INT16_C( 28438), -INT16_C( 17907),  INT16_C(     0), -INT16_C(  3615),  INT16_C(     0),
        -INT16_C( 23804),  INT16_C( 24770),  INT16_C(     0),  INT16_C( 25697), -INT16_C( 11723),  INT16_C(     0),  INT16_C(     0), -INT16_C( 19762),
        -INT16_C( 19806),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 28183),  INT16_C(     0),
         INT16_C(     0), -INT16_C( 16252), -INT16_C(  6014),  INT16_C(     0),  INT16_C(     0), -INT16_C(  1831),  INT16_C(     0), -INT16_C( 10858) } },
    { UINT32_C(3915657660),
      {  INT16_C(   626), -INT16_C( 15704), -INT16_C( 11843), -INT16_C( 25890),  INT16_C(  2650), -INT16_C( 20800), -INT16_C(  2787), -INT16_C(  8740),
        -INT16_C(  8905),  INT16_C( 13223), -INT16_C(  6849), -INT16_C( 19223),  INT16_C( 13283), -INT16_C( 24818),  INT16_C( 29285), -INT16_C( 10360),
         INT16_C( 12404),  INT16_C( 12953),  INT16_C( 30465),  INT16_C( 23756), -INT16_C( 29311), -INT16_C( 24822), -INT16_C(  6526), -INT16_C( 18052),
         INT16_C(  9411),  INT16_C(   748), -INT16_C( 10999), -INT16_C(  4682), -INT16_C( 15352),  INT16_C( 28044),  INT16_C(  5431), -INT16_C( 21692) },
      { -INT16_C(  8891),  INT16_C( 18397), -INT16_C( 21932), -INT16_C( 10589), -INT16_C( 21193), -INT16_C( 18059), -INT16_C(  3693),  INT16_C( 22130),
         INT16_C( 24085),  INT16_C(  8024),  INT16_C(  3635),  INT16_C( 15116), -INT16_C( 26414),  INT16_C(  2473), -INT16_C(  4691), -INT16_C(  3147),
        -INT16_C( 27957),  INT16_C(  7994), -INT16_C(  8900),  INT16_C( 29685),  INT16_C( 27274),  INT16_C(  7468), -INT16_C( 24996),  INT16_C( 29043),
        -INT16_C( 13316),  INT16_C( 12176), -INT16_C( 25383), -INT16_C( 21653),  INT16_C(  5173), -INT16_C(  7499),  INT16_C( 27137), -INT16_C( 13099) },
       INT32_C(          60),
      {  INT16_C(     0),  INT16_C(     0), -INT16_C(  9563), -INT16_C(  4758), -INT16_C( 21805),  INT16_C(  2967),  INT16_C(     0), -INT16_C( 15001),
         INT16_C( 30177),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 14733), -INT16_C(  8038),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(  7635),  INT16_C(     0),  INT16_C(     0), -INT16_C( 24110),  INT16_C( 10725),  INT16_C(     0),
         INT16_C( 15551),  INT16_C(     0),  INT16_C(     0),  INT16_C( 27318),  INT16_C(     0), -INT16_C( 12757),  INT16_C( 30368),  INT16_C( 19661) } },
    { UINT32_C(1991041516),
      { -INT16_C(  9908), -INT16_C( 22381),  INT16_C(  1655),  INT16_C( 29721), -INT16_C( 21807), -INT16_C( 21853),  INT16_C(  3654),  INT16_C( 31574),
         INT16_C(  2850),  INT16_C(  9310),  INT16_C( 13173),  INT16_C( 29168), -INT16_C(  9149),  INT16_C( 12202),  INT16_C( 22462),  INT16_C(  2726),
         INT16_C( 14640), -INT16_C( 22606), -INT16_C( 13504),  INT16_C(  4379), -INT16_C( 16523), -INT16_C( 17220),  INT16_C(  4813), -INT16_C(  4041),
        -INT16_C( 27363), -INT16_C( 28140),  INT16_C(  1225),  INT16_C(  3075), -INT16_C( 20767), -INT16_C( 24773), -INT16_C(  7931),  INT16_C( 13737) },
      {  INT16_C( 23323),  INT16_C( 23516), -INT16_C(  2010), -INT16_C( 25492),  INT16_C( 10423), -INT16_C( 31656), -INT16_C( 28870),  INT16_C( 22388),
        -INT16_C( 30683), -INT16_C(  4375), -INT16_C(  4723),  INT16_C( 28410),  INT16_C( 13723), -INT16_C( 24563), -INT16_C( 18921),  INT16_C( 13013),
        -INT16_C( 20207),  INT16_C( 14221), -INT16_C(  1623),  INT16_C( 24787),  INT16_C( 11042),  INT16_C( 23781),  INT16_C( 22971), -INT16_C(  8012),
        -INT16_C( 25118),  INT16_C( 28622), -INT16_C( 14198),  INT16_C(  9693), -INT16_C(  5379),  INT16_C(  5317), -INT16_C( 25952), -INT16_C( 20154) },
       INT32_C(          12),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C( 32642), -INT16_C( 26170),  INT16_C(     0),  INT16_C( 14405),  INT16_C( 26867),  INT16_C( 25975),
         INT16_C( 10370),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 22016), -INT16_C(  5279),  INT16_C( 25389),
         INT16_C(     0),  INT16_C(     0),  INT16_C(  3994), -INT16_C( 18931),  INT16_C(     0), -INT16_C( 14898),  INT16_C(     0),  INT16_C( 32267),
         INT16_C(     0),  INT16_C( 18172), -INT16_C( 25464),  INT16_C(     0),  INT16_C(  7855), -INT16_C( 20148),  INT16_C( 22954),  INT16_C(     0) } },
    { UINT32_C(4015439053),
      {  INT16_C( 15335), -INT16_C( 23989), -INT16_C(   108),  INT16_C( 30338),  INT16_C( 20637),  INT16_C( 10213), -INT16_C( 15848),  INT16_C(  5709),
         INT16_C(  4780),  INT16_C( 19498),  INT16_C( 29101), -INT16_C(  1539), -INT16_C(  6588),  INT16_C(  4590),  INT16_C( 17570), -INT16_C( 30464),
         INT16_C( 19583),  INT16_C(  5164), -INT16_C( 20917), -INT16_C(  6006),  INT16_C( 28927),  INT16_C(  5904),  INT16_C( 23858), -INT16_C(  8403),
         INT16_C( 22639),  INT16_C(  7211),  INT16_C( 10697),  INT16_C(  3349),  INT16_C(  1039), -INT16_C( 20193),  INT16_C(  8008), -INT16_C( 14278) },
      {  INT16_C( 26219), -INT16_C( 18468),  INT16_C( 26133),  INT16_C(  5279), -INT16_C( 20522),  INT16_C(  2347),  INT16_C( 22796),  INT16_C( 31976),
         INT16_C(  5041),  INT16_C( 31384), -INT16_C( 20932),  INT16_C( 19335), -INT16_C( 22862), -INT16_C(  1284),  INT16_C( 14278),  INT16_C( 12738),
        -INT16_C( 24931), -INT16_C( 19736), -INT16_C( 30715), -INT16_C(  9274), -INT16_C(  3529),  INT16_C( 17636), -INT16_C( 13237), -INT16_C(   832),
         INT16_C( 22752),  INT16_C(  7286), -INT16_C(   762), -INT16_C( 18328),  INT16_C( 25764),  INT16_C( 27315),  INT16_C( 30107),  INT16_C( 14747) },
       INT32_C(          20),
      { -INT16_C( 16778),  INT16_C(     0), -INT16_C(  1722),  INT16_C( 26657),  INT16_C(     0),  INT16_C(     0),  INT16_C(  8581),  INT16_C( 25815),
         INT16_C(     0),  INT16_C(     0),  INT16_C(  6874), -INT16_C( 24620),  INT16_C( 25674),  INT16_C(  7919),  INT16_C(     0), -INT16_C( 28669),
         INT16_C(     0),  INT16_C( 17099), -INT16_C(  6984),  INT16_C(     0),  INT16_C(  4095),  INT16_C(     0), -INT16_C( 11476),  INT16_C(     0),
        -INT16_C( 30987), -INT16_C( 15695), -INT16_C( 25441), -INT16_C( 11941),  INT16_C(     0),  INT16_C(  4598), -INT16_C(  2937), -INT16_C( 31837) } },
    { UINT32_C(1140109836),
      { -INT16_C(  9820), -INT16_C(  4217),  INT16_C( 18341), -INT16_C( 31253),  INT16_C( 24992), -INT16_C( 22878),  INT16_C(  2654),  INT16_C(   607),
         INT16_C(  4718),  INT16_C(  2668),  INT16_C(  2183), -INT16_C( 25789),  INT16_C( 11916), -INT16_C( 26444), -INT16_C( 22048), -INT16_C( 31525),
         INT16_C( 25474),  INT16_C( 10099),  INT16_C( 24234),  INT16_C( 19117),  INT16_C( 20415),  INT16_C(  7921),  INT16_C( 20569), -INT16_C( 14560),
        -INT16_C( 29342), -INT16_C(  5679),  INT16_C(  5269),  INT16_C(  8581),  INT16_C( 14659),  INT16_C(  9145), -INT16_C( 27422),  INT16_C( 25768) },
      {  INT16_C(  7159), -INT16_C( 23924),  INT16_C( 14714),  INT16_C( 14828), -INT16_C(  8824), -INT16_C(  7849),  INT16_C( 30765), -INT16_C( 28760),
         INT16_C( 31237), -INT16_C( 25991), -INT16_C(   370), -INT16_C( 11845),  INT16_C( 29751),  INT16_C(  6901), -INT16_C( 25336),  INT16_C(   126),
         INT16_C(  2744),  INT16_C( 12962), -INT16_C( 29117), -INT16_C( 13460), -INT16_C( 15508), -INT16_C( 26196),  INT16_C( 21819),  INT16_C( 16425),
        -INT16_C( 23857),  INT16_C( 24026), -INT16_C( 27232), -INT16_C( 10449),  INT16_C(  9225),  INT16_C(  4849),  INT16_C( 28865),  INT16_C( 30994) },
       INT32_C(          58),
      {  INT16_C(     0),  INT16_C(     0), -INT16_C( 27419), -INT16_C( 21273),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0), -INT16_C( 19863),  INT16_C(     0),  INT16_C(     0),  INT16_C( 12752), -INT16_C( 12181),  INT16_C(     0),  INT16_C( 27649),
         INT16_C(     0),  INT16_C(     0), -INT16_C( 21959),  INT16_C(     0), -INT16_C(   243), -INT16_C( 14746),  INT16_C( 25940), -INT16_C( 32512),
        -INT16_C( 30069),  INT16_C( 17783),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C( 30269),  INT16_C(     0) } },
    { UINT32_C(2928220226),
      {  INT16_C( 14043),  INT16_C(  5960),  INT16_C( 29067),  INT16_C( 23127),  INT16_C( 12819), -INT16_C( 19529), -INT16_C(  6457), -INT16_C( 11894),
         INT16_C( 31754), -INT16_C( 13341), -INT16_C(  2580),  INT16_C( 26181), -INT16_C(  3671), -INT16_C(  5340), -INT16_C( 20983), -INT16_C(  7014),
        -INT16_C(  7452),  INT16_C( 28667),  INT16_C( 21331),  INT16_C( 26313), -INT16_C( 32635),  INT16_C( 19481), -INT16_C( 23705),  INT16_C( 28957),
         INT16_C(    31),  INT16_C(  2877), -INT16_C( 32011), -INT16_C( 24974), -INT16_C( 27021),  INT16_C( 31882),  INT16_C(  9284),  INT16_C( 10336) },
      {  INT16_C( 23558),  INT16_C( 22935),  INT16_C( 24751),  INT16_C( 13503), -INT16_C( 10015),  INT16_C( 18560), -INT16_C( 24965), -INT16_C( 25671),
        -INT16_C(  2402), -INT16_C( 27482),  INT16_C(  6264), -INT16_C(  5326), -INT16_C( 17233), -INT16_C(  3225), -INT16_C( 14112), -INT16_C(  6628),
        -INT16_C( 19676), -INT16_C( 11457), -INT16_C(   492), -INT16_C(  2809), -INT16_C( 30762),  INT16_C( 21053), -INT16_C(  2523), -INT16_C( 15123),
        -INT16_C( 27667),  INT16_C( 25944), -INT16_C( 30036),  INT16_C( 23377), -INT16_C( 18361),  INT16_C( 10062),  INT16_C( 27264), -INT16_C( 23538) },
       INT32_C(          30),
      {  INT16_C(     0),  INT16_C(  5733),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  6242),  INT16_C(     0),
         INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 31436),  INT16_C( 28459),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),
         INT16_C( 11465),  INT16_C(     0),  INT16_C(     0),  INT16_C( 32065),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0),  INT16_C( 28987),
         INT16_C(     0),  INT16_C( 22870),  INT16_C( 25259), -INT16_C( 26924),  INT16_C(     0), -INT16_C( 30253),  INT16_C(     0),  INT16_C( 10499) } },
    { UINT32_C( 573013580),
      {  INT16_C( 25606),  INT16_C( 11124),  INT16_C( 24922),  INT16_C( 18415),  INT16_C( 18421), -INT16_C( 24147), -INT16_C(   302),  INT16_C(  6652),
         INT16_C( 19126),  INT16_C( 14144),  INT16_C( 20149), -INT16_C( 11301),  INT16_C( 21404), -INT16_C(  6139),  INT16_C( 11473), -INT16_C( 10486),
         INT16_C( 32656), -INT16_C(  5629), -INT16_C(  3360), -INT16_C( 10958), -INT16_C(  8390),  INT16_C(  3190),  INT16_C( 29405), -INT16_C( 27867),
         INT16_C( 26045),  INT16_C( 29386), -INT16_C( 22860),  INT16_C( 20549),  INT16_C( 19193), -INT16_C( 13768),  INT16_C( 17014),  INT16_C(  1698) },
      { -INT16_C( 23103), -INT16_C( 23824),  INT16_C(  8855), -INT16_C( 11913), -INT16_C(  4607), -INT16_C(  8483),  INT16_C(   608),  INT16_C(  7538),
         INT16_C( 15464),  INT16_C(  7311), -INT16_C( 11038), -INT16_C(  9364), -INT16_C( 23522), -INT16_C( 27482),  INT16_C( 18662), -INT16_C( 22374),
        -INT16_C( 29715), -INT16_C( 31670), -INT16_C( 15955), -INT16_C( 20650),  INT16_C( 13231),  INT16_C(  4237), -INT16_C(   202), -INT16_C( 25043),
        -INT16_C( 17092),  INT16_C(  7866),  INT16_C(  9873), -INT16_C( 20230), -INT16_C( 24374), -INT16_C( 20412), -INT16_C(  8216), -INT16_C( 10920) },
       INT32_C(          42),
      {  INT16_C(     0),  INT16_C(     0),  INT16_C( 26762), -INT16_C( 16571),  INT16_C(     0),  INT16_C(     0),  INT16_C( 18441),  INT16_C(     0),
         INT16_C(     0),  INT16_C(   114), -INT16_C( 10413),  INT16_C( 28525),  INT16_C( 29328),  INT16_C(  5714),  INT16_C( 17699),  INT16_C(     0),
         INT16_C( 16943),  INT16_C(  3601), -INT16_C( 31994),  INT16_C(     0),  INT16_C(     0), -INT16_C( 10174),  INT16_C(     0),  INT16_C(     0),
         INT16_C(     0),  INT16_C( 10362),  INT16_C(     0),  INT16_C(     0),  INT16_C(     0), -INT16_C(  7487),  INT16_C(     0),  INT16_C(     0) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask32 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    int       imm8 = test_vec[i].imm8;
    easysimd__m512i r = easysimd_mm512_loadu_si512(test_vec[i].r);
    easysimd__m512i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      EASYSIMD_CONSTIFY_64_(easysimd_mm512_maskz_shldi_epi16, ret, easysimd_mm512_setzero_si512(), imm8, k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_maskz_shldi_epi16");
    easysimd_assert_m512i_i16(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask32 k = easysimd_test_x86_random_mmask32();
    easysimd__m512i a = easysimd_test_x86_random_i16x32();
    easysimd__m512i b = easysimd_test_x86_random_i16x32();
    int       imm8 = easysimd_test_codegen_random_i32() & 63;
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_64_(easysimd_mm512_maskz_shldi_epi16, r, easysimd_mm512_setzero_si512(), imm8, k, a, b);

    easysimd_test_x86_write_mmask32(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i16x32(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i16x32(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_shldi_epi32(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask16 k;
    int32_t    a[16];
    int32_t    b[16];
    int        imm8;
    int32_t    r[16];
  } test_vec[8] = {
    { UINT16_C(36016),
      {  INT32_C(  1632069916),  INT32_C(    19462417),  INT32_C(  1986157335), -INT32_C(  1220746369),  INT32_C(   293292168), -INT32_C(  1478355290),  INT32_C(  1806650524), -INT32_C(  1627955071),
        -INT32_C(   620806455),  INT32_C(  1339762488),  INT32_C(    29703810), -INT32_C(  1447558623), -INT32_C(   256166838), -INT32_C(   426271414),  INT32_C(  1800488937),  INT32_C(  1879656615) },
      { -INT32_C(  1085667193), -INT32_C(  1324473041), -INT32_C(  2051812509),  INT32_C(   539913173), -INT32_C(   384767585),  INT32_C(  1892657286), -INT32_C(  1764024081), -INT32_C(   267983767),
         INT32_C(   464474348), -INT32_C(   640893579),  INT32_C(  1717469072), -INT32_C(  1970893589), -INT32_C(    59468170),  INT32_C(   762069822), -INT32_C(   842840220),  INT32_C(   398313771) },
       INT32_C(          25),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   298983891),  INT32_C(  1289854801),  INT32_C(           0),  INT32_C(    65015240),
         INT32_C(           0),  INT32_C(           0),  INT32_C(    80526591),  INT32_C(  1125453081),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1311734674) } },
    { UINT16_C(65321),
      { -INT32_C(   964773528), -INT32_C(  1521325793),  INT32_C(  1748814068),  INT32_C(   833321669),  INT32_C(  1803095976), -INT32_C(  1724537636), -INT32_C(   301642052),  INT32_C(  1173171933),
         INT32_C(   118189288), -INT32_C(   911384875), -INT32_C(   349050586),  INT32_C(   186506595), -INT32_C(   914909460), -INT32_C(   161305542), -INT32_C(   656054022),  INT32_C(  2115883670) },
      {  INT32_C(   344336702), -INT32_C(  1378012281),  INT32_C(  2140737308), -INT32_C(   661932308), -INT32_C(  2036268468), -INT32_C(  1468201810),  INT32_C(    41968236),  INT32_C(  1937874484),
         INT32_C(  1317472199),  INT32_C(  1459315770),  INT32_C(  1607832691), -INT32_C(  1774690230),  INT32_C(   270391650),  INT32_C(  1236900573),  INT32_C(   810301948), -INT32_C(  1633432105) },
       INT32_C(          20),
      { -INT32_C(  1769912222),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1403156293),  INT32_C(           0),  INT32_C(  1305118672),  INT32_C(           0),  INT32_C(           0),
        -INT32_C(   830150544), -INT32_C(   313167946), -INT32_C(  1838809767), -INT32_C(   700882042),  INT32_C(  1858142685), -INT32_C(  1012622439), -INT32_C(  1885141821),  INT32_C(   694807100) } },
    { UINT16_C(59535),
      {  INT32_C(   964428388), -INT32_C(  1701132446),  INT32_C(  2054421597), -INT32_C(   971747059), -INT32_C(   419491686), -INT32_C(   457975999),  INT32_C(  1645184885), -INT32_C(   112550251),
         INT32_C(    36947616), -INT32_C(   342045299), -INT32_C(   698019639), -INT32_C(    73631135), -INT32_C(   874341238), -INT32_C(   391080589), -INT32_C(  1303724259), -INT32_C(    39021475) },
      { -INT32_C(   402596005),  INT32_C(  1976802476),  INT32_C(   223099308),  INT32_C(  1024059571), -INT32_C(   150344828), -INT32_C(  1646216832), -INT32_C(   716166536),  INT32_C(   450034879),
        -INT32_C(  2029858085),  INT32_C(   469554799), -INT32_C(  1037481969), -INT32_C(  1241566671), -INT32_C(  1649604579),  INT32_C(   960204225),  INT32_C(  1980664759),  INT32_C(  1653662087) },
       INT32_C(          51),
      {  INT32_C(   321339398),  INT32_C(   991145628), -INT32_C(   488084895), -INT32_C(  2006325169),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   190261609),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   754077703),  INT32_C(           0), -INT32_C(  1415984676), -INT32_C(   118771596), -INT32_C(  1561652089) } },
    { UINT16_C(58729),
      {  INT32_C(  1731098686),  INT32_C(   966352698), -INT32_C(  1052592620), -INT32_C(  1907489965),  INT32_C(  1259931197),  INT32_C(   221028476), -INT32_C(   341778430),  INT32_C(  1087441154),
        -INT32_C(  1146617983),  INT32_C(  1928609886),  INT32_C(  1244935926),  INT32_C(  2010677818),  INT32_C(    96662152), -INT32_C(  1793921134), -INT32_C(   780094769),  INT32_C(  1007833531) },
      { -INT32_C(  1359496880), -INT32_C(   266212358),  INT32_C(  1547326754),  INT32_C(  1624445911), -INT32_C(  1754950395),  INT32_C(  1395423108), -INT32_C(   467292887),  INT32_C(  1327511550),
        -INT32_C(   352512016),  INT32_C(   618339842),  INT32_C(  1266685299),  INT32_C(   766202664),  INT32_C(  1841631465), -INT32_C(  1329467001), -INT32_C(  1651186018),  INT32_C(   233616413) },
       INT32_C(          11),
      {  INT32_C(  1942091127),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1880791814),  INT32_C(           0),  INT32_C(  1694753433),  INT32_C(   117446433),  INT32_C(           0),
         INT32_C(  1073483607),  INT32_C(           0), -INT32_C(  1581796772),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1753442938),  INT32_C(    93748460), -INT32_C(  1836197777) } },
    { UINT16_C(54024),
      {  INT32_C(  1927838706),  INT32_C(  1908806086),  INT32_C(    58830654), -INT32_C(   571144164),  INT32_C(  1287885752),  INT32_C(   469819440), -INT32_C(   435762194), -INT32_C(  1950806375),
         INT32_C(  1358799241), -INT32_C(   238959694), -INT32_C(  1879817358),  INT32_C(    74246475), -INT32_C(  1387253635), -INT32_C(    20295408), -INT32_C(  1243295716),  INT32_C(  1732287966) },
      { -INT32_C(   239649217),  INT32_C(  1960999169),  INT32_C(   117692348),  INT32_C(  1024159680), -INT32_C(  1326752609), -INT32_C(   911297107),  INT32_C(  1669305221),  INT32_C(  1875558192),
        -INT32_C(    10386691), -INT32_C(  1217182725), -INT32_C(   608274918), -INT32_C(  2061972763), -INT32_C(   751500506),  INT32_C(  1033692088), -INT32_C(  1482679434), -INT32_C(   669619237) },
       INT32_C(          45),
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1593604191),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
        -INT32_C(  1271840788),  INT32_C(   947279598),  INT32_C(           0),  INT32_C(           0),  INT32_C(   101694054),  INT32_C(           0), -INT32_C(  1711041292),  INT32_C(   331078402) } },
    { UINT16_C(19131),
      {  INT32_C(  1572918687), -INT32_C(   920148304),  INT32_C(  1624001835), -INT32_C(  1119648480),  INT32_C(  1675147970),  INT32_C(  2010035041),  INT32_C(  1676655500),  INT32_C(  1118743203),
         INT32_C(   815754879),  INT32_C(  1073334036),  INT32_C(   899663124),  INT32_C(   234021706), -INT32_C(    26162531),  INT32_C(   175455870), -INT32_C(  1670552327), -INT32_C(  1898046706) },
      { -INT32_C(  1631683190),  INT32_C(  1507768132), -INT32_C(   946963076), -INT32_C(    36405152), -INT32_C(   923057078),  INT32_C(  2077389186), -INT32_C(   468238123), -INT32_C(   445450917),
        -INT32_C(  1216073614),  INT32_C(  1662018279),  INT32_C(  1076535007),  INT32_C(  1748893214), -INT32_C(  1003407038),  INT32_C(  2134836138), -INT32_C(  1620879805), -INT32_C(  1098590901) },
       INT32_C(           5),
      { -INT32_C(  1206209549),  INT32_C(   620025355),  INT32_C(           0), -INT32_C(  1469012961),  INT32_C(  2065127513), -INT32_C(   103388113),  INT32_C(           0),  INT32_C(  1440044156),
         INT32_C(           0), -INT32_C(    13049204),  INT32_C(           0), -INT32_C(  1101239987),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1918066893),  INT32_C(           0) } },
    { UINT16_C(34154),
      {  INT32_C(  2049133136), -INT32_C(   948354678), -INT32_C(   637486167), -INT32_C(  1075926400), -INT32_C(  1927929558),  INT32_C(  1164140992), -INT32_C(  1806866401), -INT32_C(  1525041067),
        -INT32_C(  1944110078),  INT32_C(   659789949), -INT32_C(   738110380),  INT32_C(   680779774), -INT32_C(  1044993791),  INT32_C(   705042699), -INT32_C(   692104063),  INT32_C(   242997260) },
      { -INT32_C(  1835361515), -INT32_C(  2017858253),  INT32_C(  1079753537), -INT32_C(  1687621734), -INT32_C(  1554178408), -INT32_C(  1177656521), -INT32_C(  1014002249),  INT32_C(  2060520293),
        -INT32_C(   653431898), -INT32_C(  1704868263),  INT32_C(   484097153),  INT32_C(  1152861099), -INT32_C(  1712909215),  INT32_C(   793949560), -INT32_C(  1460412094), -INT32_C(  1843215124) },
       INT32_C(          48),
      {  INT32_C(           0),  INT32_C(  1099597753),  INT32_C(           0), -INT32_C(  1434412184),  INT32_C(           0),  INT32_C(  1640020430),  INT32_C(  1746912143),  INT32_C(           0),
         INT32_C(  1006819597),  INT32_C(           0),  INT32_C(  1414798554),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   670264798) } },
    { UINT16_C(52726),
      { -INT32_C(    24545501),  INT32_C(  1262564755),  INT32_C(  1616880249), -INT32_C(  1927882949), -INT32_C(    93366265),  INT32_C(   599677696),  INT32_C(  1515384558), -INT32_C(  1691924361),
         INT32_C(  1402581184),  INT32_C(  1587469029), -INT32_C(  1195377028), -INT32_C(   599403051),  INT32_C(   785822765), -INT32_C(    28207856), -INT32_C(    78077053), -INT32_C(  1399422996) },
      {  INT32_C(   369045295), -INT32_C(  2056020471),  INT32_C(  1899836060),  INT32_C(   894337799),  INT32_C(  1197679671),  INT32_C(  1027978425),  INT32_C(  1127783767),  INT32_C(  1290784284),
         INT32_C(   107081725),  INT32_C(   697029773),  INT32_C(   245025031), -INT32_C(  2092701620), -INT32_C(   959732211), -INT32_C(  1308422054), -INT32_C(   889898066),  INT32_C(   102163721) },
       INT32_C(          20),
      {  INT32_C(           0),  INT32_C(  1496864569),  INT32_C(   664212435),  INT32_C(           0), -INT32_C(  2139851214), -INT32_C(  1341926309), -INT32_C(   287034487), -INT32_C(  2022387972),
         INT32_C(   201352734),  INT32_C(           0), -INT32_C(   406787668),  INT32_C(  1566061630),  INT32_C(           0),  INT32_C(           0),  INT32_C(   943501139), -INT32_C(    20946578) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask16 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    int       imm8 = test_vec[i].imm8;
    easysimd__m512i r = easysimd_mm512_loadu_si512(test_vec[i].r);
    easysimd__m512i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      EASYSIMD_CONSTIFY_64_(easysimd_mm512_maskz_shldi_epi32, ret, easysimd_mm512_setzero_si512(), imm8, k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_maskz_shldi_epi32");
    easysimd_assert_m512i_i32(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    int       imm8 = easysimd_test_codegen_random_i32() & 63;
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_64_(easysimd_mm512_maskz_shldi_epi32, r, easysimd_mm512_setzero_si512(), imm8, k, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_shldi_epi64(EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  const struct {
    easysimd__mmask8 k;
    int64_t    a[8];
    int64_t    b[8];
    int        imm8;
    int64_t    r[8];
  } test_vec[8] = {
    { UINT8_C(158),
      {  INT64_C( 5542783927802503332),  INT64_C( 2203035747772250813),  INT64_C( 6534299495132079521),  INT64_C( 8818752680246665105),
         INT64_C( 8433819027969068064), -INT64_C( 2296787160820712021), -INT64_C( 8290610166774661935), -INT64_C( 5582858723219467980) },
      {  INT64_C( 2858794537398078335),  INT64_C( 4978528672276891640),  INT64_C( 5209763258467827397), -INT64_C( 3593070617478062882),
         INT64_C(  362735542682882916),  INT64_C( 4733656472905069305),  INT64_C( 4030243763975720338),  INT64_C( 9153506985934013161) },
       INT32_C(          29),
      {  INT64_C(                   0), -INT64_C( 1968504768739219154), -INT64_C( 6274758154307135102), -INT64_C( 2845135929612149975),
         INT64_C( 1711607019370255937),  INT64_C(                   0),  INT64_C(                   0),  INT64_C( 4713582324642543939) } },
    { UINT8_C(191),
      {  INT64_C(    6232773253165161), -INT64_C( 6504261809551026218),  INT64_C( 7314910789579568166), -INT64_C( 3703316361653386289),
         INT64_C( 5641179063350940573),  INT64_C( 4238851603440982997), -INT64_C(  404719050424163743), -INT64_C( 1199802126841797335) },
      {  INT64_C( 1480420106348052677),  INT64_C(  840789577223519325), -INT64_C( 7161731163042067080), -INT64_C( 6137980893275786352),
        -INT64_C( 1615257848325338814),  INT64_C( 7234194889126893488),  INT64_C( 7593812077425684050),  INT64_C( 5811363215544388374) },
       INT32_C(          37),
      {  INT64_C( 8578808782307477447),  INT64_C( 8235670577405408590),  INT64_C( 1296619786977767028), -INT64_C( 7475841286747956036),
         INT64_C( 3236526538655428976), -INT64_C( 3009677044454736501),  INT64_C(                   0), -INT64_C( 7294523896277794369) } },
    { UINT8_C(  1),
      { -INT64_C(  492849540585383322),  INT64_C( 4087035997232795249), -INT64_C( 8643170277853814412), -INT64_C( 7746087805290455568),
        -INT64_C( 5843989797570268680), -INT64_C( 5307978960543600248), -INT64_C( 2280294826509949594),  INT64_C( 6837271715284649760) },
      {  INT64_C( 3961813818086765328),  INT64_C( 3700456744528611425), -INT64_C(  928199112308486066),  INT64_C( 7722304863232687206),
         INT64_C(  629012625744230102),  INT64_C(  376902377009843515),  INT64_C(  422882754477375511),  INT64_C( 1847880212303454174) },
       INT32_C(          51),
      { -INT64_C( 3228597312973804963),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 15),
      { -INT64_C( 3162561796199800489), -INT64_C( 4148706523491917427),  INT64_C( 4618121751298339251),  INT64_C( 2660727763940618113),
         INT64_C( 1553923878140065518),  INT64_C(  356433426773081031), -INT64_C(  701174559426504768), -INT64_C( 5544282139277396196) },
      {  INT64_C(  330169312300996598), -INT64_C( 8805061988566665350), -INT64_C( 5826305990180887111), -INT64_C( 5891650905289721916),
         INT64_C( 2630033645362850622), -INT64_C( 8393656570865602760), -INT64_C( 7015238391336203166),  INT64_C( 2394335980860947668) },
       INT32_C(          14),
      {  INT64_C( 1491634112599277861),  INT64_C( 3844230728122589555), -INT64_C( 5237417084590429239),  INT64_C( 3707438227416705935),
         INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 40),
      { -INT64_C( 8675234932615685093), -INT64_C( 4843375246341167390), -INT64_C( 1107290063738149245),  INT64_C( 8855966239613174615),
        -INT64_C( 8510430390689445998),  INT64_C( 3504434575687863814), -INT64_C( 6402550422054095961),  INT64_C( 7795815457208214358) },
      {  INT64_C( 6612320354696787091),  INT64_C( 2475541063322909564),  INT64_C( 3679296710948375063),  INT64_C( 6117604358787462303),
        -INT64_C( 2721210611180589171),  INT64_C( 8658837004395773324), -INT64_C( 7153423760444390999),  INT64_C( 5149862575304253795) },
       INT32_C(          43),
      {  INT64_C(                   0),  INT64_C(                   0),  INT64_C(                   0), -INT64_C(  713052363744214065),
         INT64_C(                   0), -INT64_C( 4147758351394863048),  INT64_C(                   0),  INT64_C(                   0) } },
    { UINT8_C( 86),
      {  INT64_C( 4991091093721506661), -INT64_C( 8533630132329465068),  INT64_C( 4650265977968930349),  INT64_C( 6765521513122418823),
        -INT64_C( 4322184947222657320), -INT64_C( 7682037708086447297), -INT64_C( 1756705806471639015),  INT64_C( 6274456011385637565) },
      {  INT64_C( 1554335430790763294), -INT64_C( 1222072168310904050),  INT64_C( 1795507277102149283), -INT64_C( 3076554790188195978),
        -INT64_C( 8277557295351372748),  INT64_C(  424760458541666080), -INT64_C( 7642622724547267717), -INT64_C( 3251922608126706350) },
       INT32_C(          61),
      {  INT64_C(                   0), -INT64_C( 7070288048679944863), -INT64_C( 6693090618003313196),  INT64_C(                   0),
         INT64_C( 1271148347294772358),  INT64_C(                   0),  INT64_C( 3656358177858979439),  INT64_C(                   0) } },
    { UINT8_C(182),
      {  INT64_C( 6372536620462524925),  INT64_C( 8071483514376451249), -INT64_C(  146706580874552934), -INT64_C( 6638034657210611287),
         INT64_C( 3738865661685773887), -INT64_C(  180113877104019565), -INT64_C( 5271943790937685324),  INT64_C( 5142477488959525140) },
      { -INT64_C( 4743498582041948810),  INT64_C( 1426601355384865645), -INT64_C( 9127737144181513840),  INT64_C( 7575786795824853972),
         INT64_C( 1698012824332268710),  INT64_C( 4486627210205487056), -INT64_C( 4124400116207767596),  INT64_C( 4593606303508791138) },
       INT32_C(          57),
      {  INT64_C(                   0),  INT64_C( 7072789538805881990),  INT64_C( 3819799631609190467),  INT64_C(                   0),
         INT64_C( 9092522573969015785),  INT64_C( 2773240348520991935),  INT64_C(                   0),  INT64_C( 2918191310763279870) } },
    { UINT8_C(174),
      {  INT64_C( 1418262660016667297), -INT64_C(  715181980594410255),  INT64_C( 4649033662117566956),  INT64_C( 8448010544035251054),
         INT64_C( 6578110244326915286),  INT64_C( 5246287141041549840),  INT64_C( 4909970152326637416),  INT64_C( 5675680638809939066) },
      {  INT64_C( 7967658694337182413),  INT64_C(  334413834616114689),  INT64_C(  744514022491609146),  INT64_C(  643764309861083432),
         INT64_C( 4909717732015961174), -INT64_C( 5295875290136811296), -INT64_C( 7154500770102613731), -INT64_C( 9041209161593057468) },
       INT32_C(          32),
      {  INT64_C(                   0),  INT64_C( 2371405723719701404), -INT64_C( 1487918594060383345), -INT64_C( 8072205979721196486),
         INT64_C(                   0), -INT64_C( 2919684886056843909),  INT64_C(                   0),  INT64_C( 5680196342522522667) } }
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])); i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m512i a = easysimd_mm512_loadu_si512(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_si512(test_vec[i].b);
    int       imm8 = test_vec[i].imm8;
    easysimd__m512i r = easysimd_mm512_loadu_si512(test_vec[i].r);
    easysimd__m512i ret;
    EASYSIMD_TEST_PERF_WITH_LOOP_START(100000) {
      EASYSIMD_CONSTIFY_64_(easysimd_mm512_maskz_shldi_epi64, ret, easysimd_mm512_setzero_si512(), imm8, k, a, b);
    } EASYSIMD_TEST_PERF_END("_mm512_maskz_shldi_epi64");
    easysimd_assert_m512i_i64(r, ==, ret);
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m512i a = easysimd_test_x86_random_i64x8();
    easysimd__m512i b = easysimd_test_x86_random_i64x8();
    int       imm8 = easysimd_test_codegen_random_i32() & 63;
    easysimd__m512i r;
    EASYSIMD_CONSTIFY_64_(easysimd_mm512_maskz_shldi_epi64, r, easysimd_mm512_setzero_si512(), imm8, k, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i64x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_codegen_write_i32(2, imm8, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i64x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_shldi_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_shldi_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_shldi_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_shldi_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_shldi_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_shldi_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_shldi_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_shldi_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_shldi_epi64)

  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_shldi_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_shldi_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_shldi_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_shldi_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_shldi_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_shldi_epi64)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_shldi_epi16)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_shldi_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_shldi_epi64)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
