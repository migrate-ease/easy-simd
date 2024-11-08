#define EASYSIMD_TEST_X86_AVX512_INSN dpwssd

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/dpwssd.h>

static int
test_easysimd_mm_dpwssd_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[4];
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { { -INT32_C(   162181129),  INT32_C(  1617503567), -INT32_C(  1477930415),  INT32_C(   981415384) },
      { -INT32_C(  1038565035),  INT32_C(   289426369),  INT32_C(  2134447221),  INT32_C(   594328108) },
      {  INT32_C(  1494860042),  INT32_C(   934904806), -INT32_C(   304176619),  INT32_C(   740842967) },
      { -INT32_C(   265077087),  INT32_C(  1064382197), -INT32_C(  1755028904),  INT32_C(   654484908) } },
    { { -INT32_C(   521256929),  INT32_C(    32583307),  INT32_C(  1870670402), -INT32_C(   275583515) },
      { -INT32_C(  1790399314),  INT32_C(  1154220335),  INT32_C(  2050075555),  INT32_C(   665213192) },
      {  INT32_C(   621253786),  INT32_C(    86440130),  INT32_C(   141862691),  INT32_C(  1123485332) },
      { -INT32_C(   193776189),  INT32_C(    55234161), -INT32_C(  1865294657), -INT32_C(    63200337) } },
    { { -INT32_C(   505987150), -INT32_C(   467295168),  INT32_C(  1465800527),  INT32_C(  1249838512) },
      {  INT32_C(  1550878361), -INT32_C(  1570662785), -INT32_C(   777333443), -INT32_C(  1911250469) },
      {  INT32_C(   544205792), -INT32_C(   570124913), -INT32_C(  1674157076),  INT32_C(    31962472) },
      { -INT32_C(   149385950),  INT32_C(   477185973),  INT32_C(  1488802919),  INT32_C(  1695868340) } },
    { { -INT32_C(  1168287941),  INT32_C(   727498477),  INT32_C(  1878787731), -INT32_C(  2013458265) },
      { -INT32_C(  1951896324), -INT32_C(   311776255),  INT32_C(  2005573647), -INT32_C(  1871089323) },
      { -INT32_C(  1253321016),  INT32_C(   685811605),  INT32_C(  1436016046), -INT32_C(   371354387) },
      { -INT32_C(   893069741),  INT32_C(  1164431170), -INT32_C(  1529866197),  INT32_C(  1648032205) } },
    { {  INT32_C(    41256193),  INT32_C(  1106304817), -INT32_C(   726107521), -INT32_C(  1285279253) },
      { -INT32_C(  1654083832),  INT32_C(    96815447),  INT32_C(   324689190), -INT32_C(   218286095) },
      { -INT32_C(   285904196), -INT32_C(   802167471),  INT32_C(  1252321119),  INT32_C(   570230809) },
      { -INT32_C(   457444503),  INT32_C(   959972835), -INT32_C(   781795519), -INT32_C(  1284574851) } },
    { {  INT32_C(   264136120), -INT32_C(   736853074), -INT32_C(   756519200),  INT32_C(  1657071014) },
      { -INT32_C(  1471104681), -INT32_C(    42434658), -INT32_C(  2142823321), -INT32_C(   593411036) },
      {  INT32_C(  1491820458), -INT32_C(  1020395550),  INT32_C(   345314670),  INT32_C(  1366710778) },
      { -INT32_C(   689847314), -INT32_C(   727743166), -INT32_C(   888910987),  INT32_C(  1870043140) } },
    { { -INT32_C(  1309030637), -INT32_C(  1381011130), -INT32_C(  1322387827), -INT32_C(   443625925) },
      {  INT32_C(   272464173), -INT32_C(   405575047),  INT32_C(  2046519423), -INT32_C(   724930111) },
      {  INT32_C(  2139538232), -INT32_C(  1054067404),  INT32_C(  1718835755),  INT32_C(  1447756072) },
      { -INT32_C(  1656007975), -INT32_C(   910226914),  INT32_C(   114096035), -INT32_C(   679376751) } },
    { { -INT32_C(   211383942),  INT32_C(  1943681523),  INT32_C(  1659688353),  INT32_C(  2134357831) },
      { -INT32_C(  1342259846),  INT32_C(   493890546), -INT32_C(  1383799931),  INT32_C(  1594085348) },
      {  INT32_C(  1280469336),  INT32_C(  1136602530),  INT32_C(  1235659522), -INT32_C(   573973150) },
      { -INT32_C(  1070826298), -INT32_C(  2089166425),  INT32_C(  1420204867),  INT32_C(  2030457066) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_x_mm_loadu_epi32(test_vec[i].src);
    easysimd__m128i a = easysimd_x_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r = easysimd_mm_dpwssd_epi32(src, a, b);
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_x_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i32x4();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_dpwssd_epi32(src, a, b);

    easysimd_test_x86_write_i32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_dpwssd_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[4];
    const easysimd__mmask8 k;
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { { -INT32_C(   238369957), -INT32_C(  1461737740), -INT32_C(   267041609),  INT32_C(  1516210125) },
      UINT8_C( 83),
      {  INT32_C(  1256980857),  INT32_C(   734982924),  INT32_C(  1978883789), -INT32_C(   456032223) },
      { -INT32_C(   237452725), -INT32_C(   425164639),  INT32_C(   481532309),  INT32_C(  1903169016) },
      { -INT32_C(   313833746), -INT32_C(  1426369360), -INT32_C(   267041609),  INT32_C(  1516210125) } },
    { {  INT32_C(   448486414),  INT32_C(   474384719),  INT32_C(   160512488),  INT32_C(    82731705) },
      UINT8_C( 37),
      {  INT32_C(  1204221382),  INT32_C(   937208990),  INT32_C(  1815083359), -INT32_C(   981819287) },
      { -INT32_C(   451635876),  INT32_C(   348991707),  INT32_C(   617469889), -INT32_C(  1958096187) },
      {  INT32_C(   393324206),  INT32_C(   474384719),  INT32_C(   438911274),  INT32_C(    82731705) } },
    { {  INT32_C(  1708265415),  INT32_C(   463253436),  INT32_C(   294177704), -INT32_C(   925498772) },
      UINT8_C(151),
      {  INT32_C(   443723242),  INT32_C(  1390118523),  INT32_C(   639106900),  INT32_C(  1491968584) },
      {  INT32_C(   605311861), -INT32_C(  1144246289), -INT32_C(  1171792457), -INT32_C(  1655574605) },
      {  INT32_C(  1321829025), -INT32_C(   288829163),  INT32_C(   121329717), -INT32_C(   925498772) } },
    { {  INT32_C(   397919388), -INT32_C(  1637248438), -INT32_C(   624656238), -INT32_C(  1741508061) },
      UINT8_C(  5),
      {  INT32_C(  1995750470),  INT32_C(  1697492872), -INT32_C(   988223530), -INT32_C(    43862727) },
      { -INT32_C(    28870292),  INT32_C(  1670440675),  INT32_C(  1552313258), -INT32_C(   463397219) },
      { -INT32_C(   154444336), -INT32_C(  1637248438), -INT32_C(  1152339650), -INT32_C(  1741508061) } },
    { {  INT32_C(  1650087642), -INT32_C(   590903547), -INT32_C(  1465786513),  INT32_C(    10814356) },
      UINT8_C(125),
      { -INT32_C(   765394964), -INT32_C(    92421233),  INT32_C(  1754847562), -INT32_C(  1857848261) },
      {  INT32_C(  1570153942),  INT32_C(  1288467053),  INT32_C(   400651284),  INT32_C(   110420249) },
      {  INT32_C(  1370719842), -INT32_C(   590903547), -INT32_C(  1596590385), -INT32_C(   288518893) } },
    { {  INT32_C(  1893266656),  INT32_C(    40523192),  INT32_C(  1785332271),  INT32_C(  1425780094) },
      UINT8_C( 83),
      {  INT32_C(    79737489), -INT32_C(   216527746), -INT32_C(   821284883), -INT32_C(  1196485948) },
      {  INT32_C(  1081090027), -INT32_C(  1938853238), -INT32_C(  1978934819),  INT32_C(  1725784020) },
      {  INT32_C(  1751352059),  INT32_C(   232043788),  INT32_C(  1785332271),  INT32_C(  1425780094) } },
    { { -INT32_C(  1872061167), -INT32_C(  1752989014),  INT32_C(  1969655729), -INT32_C(  1926359390) },
      UINT8_C( 53),
      {  INT32_C(   297782686),  INT32_C(   384715837), -INT32_C(  1226082217), -INT32_C(   204975786) },
      {  INT32_C(  1033722043),  INT32_C(  1793996251), -INT32_C(  1324587877),  INT32_C(   820484498) },
      { -INT32_C(  2093382498), -INT32_C(  1752989014), -INT32_C(  1153730974), -INT32_C(  1926359390) } },
    { { -INT32_C(  1539201433),  INT32_C(  1236938738), -INT32_C(    33512024), -INT32_C(  1309554442) },
      UINT8_C( 32),
      { -INT32_C(   990122353),  INT32_C(  1096771037),  INT32_C(   198381938), -INT32_C(  1636695048) },
      {  INT32_C(  1955665477),  INT32_C(  1981602513), -INT32_C(  1569908006), -INT32_C(  1681777140) },
      { -INT32_C(  1539201433),  INT32_C(  1236938738), -INT32_C(    33512024), -INT32_C(  1309554442) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_x_mm_loadu_epi32(test_vec[i].src);
    easysimd__m128i a = easysimd_x_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r = easysimd_mm_mask_dpwssd_epi32(src, test_vec[i].k, a, b);
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_x_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_mask_dpwssd_epi32(src, k, a, b);

    easysimd_test_x86_write_i32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_maskz_dpwssd_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int32_t src[4];
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { UINT8_C( 41),
      { -INT32_C(   362824435), -INT32_C(  1792295389), -INT32_C(   945750321),  INT32_C(  1181094657) },
      { -INT32_C(   772097033), -INT32_C(   223579689), -INT32_C(    50442530), -INT32_C(   838231791) },
      { -INT32_C(  1225657352),  INT32_C(   478578206),  INT32_C(   975064359), -INT32_C(  1436326733) },
      { -INT32_C(    88821329),  INT32_C(           0),  INT32_C(           0),  INT32_C(   782132575) } },
    { UINT8_C(166),
      { -INT32_C(   142844385), -INT32_C(  1377228593), -INT32_C(  2018908298),  INT32_C(  2018947968) },
      { -INT32_C(   825938361),  INT32_C(    49668532),  INT32_C(  1689595903), -INT32_C(   259825518) },
      {  INT32_C(    79657692),  INT32_C(  1249537123), -INT32_C(   909508074), -INT32_C(  1620098472) },
      {  INT32_C(           0), -INT32_C(  1578001363),  INT32_C(  1924831538),  INT32_C(           0) } },
    { UINT8_C(119),
      {  INT32_C(   627913841),  INT32_C(   354902550),  INT32_C(   611966353),  INT32_C(   420805949) },
      {  INT32_C(  1088101844), -INT32_C(  1722407529),  INT32_C(   854663152), -INT32_C(  1012690545) },
      {  INT32_C(  1641662974), -INT32_C(   839716880),  INT32_C(  1695159912), -INT32_C(     2284757) },
      {  INT32_C(   938227052),  INT32_C(   652110322),  INT32_C(   996182395),  INT32_C(           0) } },
    { UINT8_C(108),
      { -INT32_C(   666912704), -INT32_C(   848193827),  INT32_C(  1140876213), -INT32_C(   234314764) },
      { -INT32_C(   799190047),  INT32_C(  1547250246),  INT32_C(  1485282869),  INT32_C(   865699451) },
      {  INT32_C(  1544647110),  INT32_C(  1158799074), -INT32_C(  2110171426),  INT32_C(  1072573534) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(   145578054), -INT32_C(   371472292) } },
    { UINT8_C( 13),
      { -INT32_C(   988787841), -INT32_C(  1423816330),  INT32_C(  1661184487), -INT32_C(   157901776) },
      { -INT32_C(  2031070553), -INT32_C(  1184615069), -INT32_C(  1692932497),  INT32_C(   538597333) },
      { -INT32_C(  1332289433), -INT32_C(  1416150782), -INT32_C(   505677243),  INT32_C(   955240849) },
      { -INT32_C(   529875632),  INT32_C(           0),  INT32_C(  1870139263), -INT32_C(   305403621) } },
    { UINT8_C( 25),
      { -INT32_C(  2017534173), -INT32_C(  2092948716), -INT32_C(   568436727),  INT32_C(   385825199) },
      { -INT32_C(   686045547),  INT32_C(  1478280797),  INT32_C(  1944714658), -INT32_C(   879287828) },
      {  INT32_C(    65019616), -INT32_C(  1240702114), -INT32_C(  1150948478),  INT32_C(  2144631786) },
      { -INT32_C(  2144100317),  INT32_C(           0),  INT32_C(           0),  INT32_C(   222241139) } },
    { UINT8_C(128),
      { -INT32_C(  1621692607),  INT32_C(  1475834549),  INT32_C(  1556865136), -INT32_C(   517447167) },
      {  INT32_C(  1809769480), -INT32_C(   588409359),  INT32_C(  1992730874),  INT32_C(  1807172988) },
      {  INT32_C(   237065884),  INT32_C(   763263053), -INT32_C(  1523590333),  INT32_C(   186978307) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C( 23),
      { -INT32_C(   428473099), -INT32_C(  1815976807), -INT32_C(  1995864052),  INT32_C(  1777648077) },
      { -INT32_C(  1905952747), -INT32_C(   774794506), -INT32_C(   757827647), -INT32_C(  1597513828) },
      { -INT32_C(  1170625194), -INT32_C(   137966225), -INT32_C(  1748742186),  INT32_C(  1504652868) },
      { -INT32_C(   559504512), -INT32_C(  1429730839), -INT32_C(  1060722510),  INT32_C(           0) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_x_mm_loadu_epi32(test_vec[i].src);
    easysimd__m128i a = easysimd_x_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r = easysimd_mm_maskz_dpwssd_epi32(test_vec[i].k, src, a, b);
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_x_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i32x4();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_maskz_dpwssd_epi32(k, src, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, src, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_dpwssd_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[8];
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(  1020727724), -INT32_C(   316025130),  INT32_C(   390778450), -INT32_C(   580234552), -INT32_C(   599737840), -INT32_C(   223586209),  INT32_C(  1366918768), -INT32_C(  1046428564) },
      { -INT32_C(  2004563790), -INT32_C(   243946082),  INT32_C(  1141423996),  INT32_C(   555905553), -INT32_C(  1929551315),  INT32_C(   712944058), -INT32_C(  1686309073),  INT32_C(  1834753211) },
      { -INT32_C(  2097815324),  INT32_C(   175336078),  INT32_C(   978221865),  INT32_C(   458977518), -INT32_C(  1918412590),  INT32_C(   817308929), -INT32_C(   657706211),  INT32_C(   860235855) },
      {  INT32_C(    71179056), -INT32_C(   901277175),  INT32_C(   129996702),  INT32_C(   323365804),  INT32_C(   833621253), -INT32_C(   297147349),  INT32_C(  1595632275), -INT32_C(   603056823) } },
    { { -INT32_C(  1749665015), -INT32_C(   811521626), -INT32_C(  1827999835),  INT32_C(   850290016), -INT32_C(  1077979714), -INT32_C(  1745848454), -INT32_C(    93274966), -INT32_C(   315771164) },
      { -INT32_C(  1752898575), -INT32_C(  1301863155),  INT32_C(  1967485205), -INT32_C(  1800932394), -INT32_C(  1034655928), -INT32_C(  2007350050), -INT32_C(   461190656),  INT32_C(  1909633152) },
      { -INT32_C(  1610000749), -INT32_C(  1856868228), -INT32_C(  1224304671), -INT32_C(   766726774), -INT32_C(   225075180), -INT32_C(   461639708),  INT32_C(   969473465),  INT32_C(  1101699757) },
      { -INT32_C(  1251743728),  INT32_C(    24463344),  INT32_C(  1134479544),  INT32_C(  1237554256), -INT32_C(  1673008574), -INT32_C(  1602766224), -INT32_C(   189321670),  INT32_C(   701949144) } },
    { {  INT32_C(  1843508209),  INT32_C(    83768355),  INT32_C(  1455162571), -INT32_C(   970454863), -INT32_C(  1934049880),  INT32_C(  1701852076), -INT32_C(   560056271), -INT32_C(  1004582445) },
      {  INT32_C(   540082684),  INT32_C(     2371381), -INT32_C(   464068557), -INT32_C(  1867874328), -INT32_C(   384015556), -INT32_C(   951153514), -INT32_C(  1733890619),  INT32_C(   844940598) },
      { -INT32_C(    78475834), -INT32_C(   268666948),  INT32_C(  1087590999),  INT32_C(   248544977),  INT32_C(  2012736993),  INT32_C(  1044333945),  INT32_C(  1758913842),  INT32_C(  1905996458) },
      {  INT32_C(  1818780555),  INT32_C(   454050175),  INT32_C(  1166032642), -INT32_C(    15500487),  INT32_C(  2063101800),  INT32_C(   945095876), -INT32_C(  1238111121), -INT32_C(   824842909) } },
    { {  INT32_C(  2070736319), -INT32_C(  1133877148), -INT32_C(  1929625925), -INT32_C(  1650799428),  INT32_C(   856986041),  INT32_C(   158421975), -INT32_C(   479049672),  INT32_C(   978587002) },
      {  INT32_C(  1605746938), -INT32_C(   467984343),  INT32_C(   443553630), -INT32_C(  1648948253),  INT32_C(  1943063452),  INT32_C(  1467826463),  INT32_C(    70971273), -INT32_C(   163672324) },
      {  INT32_C(  2018898767),  INT32_C(  1901883411),  INT32_C(  1804323975),  INT32_C(  1946698455),  INT32_C(   770168846), -INT32_C(  1551604711),  INT32_C(  1336393555), -INT32_C(  1673075379) },
      { -INT32_C(  1417058306), -INT32_C(  1104895965), -INT32_C(  1822120931),  INT32_C(  1948854297),  INT32_C(  1342530801),  INT32_C(    55344978), -INT32_C(   387198855),  INT32_C(  1240111162) } },
    { { -INT32_C(   334193704), -INT32_C(  1822593012),  INT32_C(   352250173),  INT32_C(   965215787), -INT32_C(   127504162),  INT32_C(   681307092), -INT32_C(   143179094), -INT32_C(     7029465) },
      {  INT32_C(  1709942873),  INT32_C(  1459112217),  INT32_C(  1567291186), -INT32_C(   594021379),  INT32_C(   936705379), -INT32_C(  1839239192), -INT32_C(   661989455), -INT32_C(   304669036) },
      { -INT32_C(   548224058),  INT32_C(  1043680012),  INT32_C(  1067163714), -INT32_C(   165989741),  INT32_C(   405663536),  INT32_C(   262901086), -INT32_C(   119065244),  INT32_C(   434552659) },
      { -INT32_C(   206506508), -INT32_C(  1108524656),  INT32_C(   796887375),  INT32_C(   948365407), -INT32_C(    36171598), -INT32_C(   271965831), -INT32_C(   265863852), -INT32_C(   167460531) } },
    { { -INT32_C(  1896269694), -INT32_C(   976474493),  INT32_C(  1627744206), -INT32_C(   900259686),  INT32_C(  1860338960),  INT32_C(  1988005138),  INT32_C(   359556546), -INT32_C(  1506847708) },
      {  INT32_C(   271853709),  INT32_C(   634781782),  INT32_C(    42392424),  INT32_C(   197975803),  INT32_C(  1970974307), -INT32_C(    18089924), -INT32_C(  2112595619),  INT32_C(   992494510) },
      { -INT32_C(  1052025493), -INT32_C(   974773923), -INT32_C(   121147908), -INT32_C(  1375497397),  INT32_C(  2116255042), -INT32_C(   746844555),  INT32_C(   357929063),  INT32_C(  1045462483) },
      { -INT32_C(  1714595923), -INT32_C(  1119809531),  INT32_C(  1362784344), -INT32_C(   728407929), -INT32_C(  2133463740),  INT32_C(  1983804242), -INT32_C(   470057263), -INT32_C(   707182834) } },
    { {  INT32_C(   939564250), -INT32_C(  1157765443), -INT32_C(  1632451245), -INT32_C(  1706248872), -INT32_C(  1474793677), -INT32_C(   444885635), -INT32_C(   101003227),  INT32_C(   674712398) },
      { -INT32_C(  1537198105),  INT32_C(  1902009886),  INT32_C(  2064650275), -INT32_C(   115974970),  INT32_C(  1235300044), -INT32_C(   382853693),  INT32_C(  1004677613),  INT32_C(  1533286772) },
      {  INT32_C(  1879032913),  INT32_C(  1172332066),  INT32_C(   885125230),  INT32_C(   422434637), -INT32_C(   916271355), -INT32_C(   642608660),  INT32_C(   773166266), -INT32_C(    24544851) },
      {  INT32_C(    48399505), -INT32_C(    58002695), -INT32_C(  1223455891), -INT32_C(  1965107772), -INT32_C(  1888557615), -INT32_C(   602302403), -INT32_C(   214904863),  INT32_C(   868905782) } },
    { {  INT32_C(  1617791037),  INT32_C(  1436896998), -INT32_C(  1953929666),  INT32_C(  1134868285),  INT32_C(  1913390982),  INT32_C(  1380695704), -INT32_C(     8363950),  INT32_C(   402524633) },
      {  INT32_C(  2021092241), -INT32_C(   120775495), -INT32_C(  1065134461), -INT32_C(  1828509683), -INT32_C(   955969745),  INT32_C(   521753037), -INT32_C(  1960863311),  INT32_C(   866262434) },
      {  INT32_C(  1118509448), -INT32_C(  1204127691), -INT32_C(   596001329),  INT32_C(   342916325),  INT32_C(  1507554700),  INT32_C(  2021258438),  INT32_C(   788764813),  INT32_C(  1029940661) },
      { -INT32_C(  1970865214),  INT32_C(  1697034149),  INT32_C(  2113577750),  INT32_C(  1316709422),  INT32_C(  1694813825),  INT32_C(  1566036471),  INT32_C(   325153068),  INT32_C(   434895113) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_x_mm256_loadu_epi32(test_vec[i].src);
    easysimd__m256i a = easysimd_x_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_x_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r = easysimd_mm256_dpwssd_epi32(src, a, b);
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_x_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i32x8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_dpwssd_epi32(src, a, b);

    easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_dpwssd_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[8];
    const easysimd__mmask8 k;
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { {  INT32_C(  1700850591),  INT32_C(  1339199166),  INT32_C(   174250865), -INT32_C(  1265025093), -INT32_C(   136720169), -INT32_C(  2109114930), -INT32_C(   795042753), -INT32_C(  1374719217) },
      UINT8_C( 94),
      { -INT32_C(    48491665), -INT32_C(  1066505243),  INT32_C(   226261453), -INT32_C(   488361966),  INT32_C(  2142296841), -INT32_C(  1044499932),  INT32_C(  1120964303),  INT32_C(   228621982) },
      {  INT32_C(  1980480657), -INT32_C(   180913881),  INT32_C(    67286002), -INT32_C(   303634717), -INT32_C(   429091135), -INT32_C(  1733874999),  INT32_C(  1457158072), -INT32_C(  2040235275) },
      {  INT32_C(  1700850591), -INT32_C(  2054170301), -INT32_C(   429301453), -INT32_C(  1309607799), -INT32_C(    95799584), -INT32_C(  2109114930), -INT32_C(  1302839513), -INT32_C(  1374719217) } },
    { {  INT32_C(  1576824630), -INT32_C(   615369752), -INT32_C(   891333402), -INT32_C(    55065030),  INT32_C(   635577180),  INT32_C(    96307533), -INT32_C(   178481408),  INT32_C(  1199292433) },
      UINT8_C( 47),
      { -INT32_C(  1424513673),  INT32_C(  1284633335), -INT32_C(  1736025134),  INT32_C(   905216530), -INT32_C(   310240668),  INT32_C(  1877838039), -INT32_C(  1535057180), -INT32_C(   707540899) },
      {  INT32_C(  1686170221), -INT32_C(  1347415587),  INT32_C(  2135373677),  INT32_C(   515193785),  INT32_C(   755709781), -INT32_C(  1549993537),  INT32_C(   960961755),  INT32_C(  1359878884) },
      {  INT32_C(  1145581665), -INT32_C(  1033626413), -INT32_C(  1420930438), -INT32_C(   439375680),  INT32_C(   635577180), -INT32_C(   532554718), -INT32_C(   178481408),  INT32_C(  1199292433) } },
    { { -INT32_C(   491352571),  INT32_C(   210855583),  INT32_C(  1468848285),  INT32_C(  1769292051),  INT32_C(   949387384),  INT32_C(  1440428665),  INT32_C(   864953166),  INT32_C(  1115987005) },
      UINT8_C( 42),
      { -INT32_C(  1580653510), -INT32_C(  1925261643), -INT32_C(  1549691550), -INT32_C(  1961162230), -INT32_C(   771468384),  INT32_C(  1377851695), -INT32_C(  2087758873),  INT32_C(   313381592) },
      { -INT32_C(  1397524490), -INT32_C(  1355156915), -INT32_C(  1840063865), -INT32_C(  2078446108), -INT32_C(   262790719),  INT32_C(  1648523131), -INT32_C(  1561931318), -INT32_C(  1699376221) },
      { -INT32_C(   491352571),  INT32_C(   855435742),  INT32_C(  1468848285), -INT32_C(  1504304630),  INT32_C(   949387384), -INT32_C(  1627369138),  INT32_C(   864953166),  INT32_C(  1115987005) } },
    { {  INT32_C(  1464231946), -INT32_C(   502890662),  INT32_C(  1047812186), -INT32_C(  2017226298),  INT32_C(   762780082),  INT32_C(  1536211344),  INT32_C(   788362890),  INT32_C(   348697097) },
      UINT8_C( 27),
      { -INT32_C(  1921684722), -INT32_C(   890808462),  INT32_C(  1552950987),  INT32_C(    34543593),  INT32_C(  1200831630),  INT32_C(  1121119948), -INT32_C(  1639186197), -INT32_C(   692494136) },
      {  INT32_C(  1029910219),  INT32_C(  1359498118),  INT32_C(  1521391729),  INT32_C(  1046265264), -INT32_C(   964235271), -INT32_C(   922199843),  INT32_C(   543642712), -INT32_C(  2131353420) },
      {  INT32_C(  1331717475), -INT32_C(   352052930),  INT32_C(  1047812186), -INT32_C(  2112723366),  INT32_C(   429570754),  INT32_C(  1536211344),  INT32_C(   788362890),  INT32_C(   348697097) } },
    { { -INT32_C(   725788338),  INT32_C(   354862500),  INT32_C(   242209886), -INT32_C(  1974678383), -INT32_C(  1722756421), -INT32_C(  2107483862),  INT32_C(  1654835629),  INT32_C(   937597161) },
      UINT8_C(242),
      {  INT32_C(  1704332447),  INT32_C(   113486898), -INT32_C(   409480933), -INT32_C(   257744611), -INT32_C(   887473038),  INT32_C(  1735957918), -INT32_C(   665789889),  INT32_C(  1556776892) },
      { -INT32_C(   977182573),  INT32_C(   684426252),  INT32_C(  1913610837), -INT32_C(   161303932), -INT32_C(  1933476370),  INT32_C(  1509112090), -INT32_C(   802077932),  INT32_C(  1580006347) },
      { -INT32_C(   725788338),  INT32_C(  1053717661),  INT32_C(   242209886), -INT32_C(  1974678383), -INT32_C(   824375983), -INT32_C(  1865713634),  INT32_C(  1613216489),  INT32_C(  1543417269) } },
    { {  INT32_C(  1747250524), -INT32_C(   963580047), -INT32_C(   700866478),  INT32_C(  1103928146),  INT32_C(   852331800), -INT32_C(   628309562), -INT32_C(   810828540),  INT32_C(   355391417) },
      UINT8_C(197),
      {  INT32_C(  1094090066), -INT32_C(  1366033138), -INT32_C(   788502218), -INT32_C(  1024835275), -INT32_C(   796320753),  INT32_C(  1725195176), -INT32_C(   434134002),  INT32_C(   615199954) },
      { -INT32_C(  1083842127),  INT32_C(   342751710), -INT32_C(  1729728926), -INT32_C(  1101344593), -INT32_C(  1785797652),  INT32_C(  1442538311), -INT32_C(   650438137), -INT32_C(      137650) },
      {  INT32_C(  1222224220), -INT32_C(   963580047),  INT32_C(   377807102),  INT32_C(  1103928146),  INT32_C(   852331800), -INT32_C(   628309562), -INT32_C(   901909109),  INT32_C(   266415540) } },
    { { -INT32_C(  1497406776), -INT32_C(  1095029669),  INT32_C(  1247191450),  INT32_C(  1560850545), -INT32_C(   604858476),  INT32_C(    19983866),  INT32_C(  1440377863),  INT32_C(   441833298) },
      UINT8_C( 57),
      {  INT32_C(  1083556116),  INT32_C(   500913020),  INT32_C(  1502487977), -INT32_C(   991040723), -INT32_C(   876689186),  INT32_C(  1708376057),  INT32_C(  1891051673), -INT32_C(  1851075971) },
      {  INT32_C(   265437075),  INT32_C(   992783762), -INT32_C(     7030062), -INT32_C(  2084273499),  INT32_C(  1129284170), -INT32_C(   609738174), -INT32_C(   934518710), -INT32_C(   983960014) },
      { -INT32_C(  1692605826), -INT32_C(  1095029669),  INT32_C(  1247191450), -INT32_C(  2082067738), -INT32_C(   381158046), -INT32_C(   366292572),  INT32_C(  1440377863),  INT32_C(   441833298) } },
    { { -INT32_C(   942396619), -INT32_C(  1442709288),  INT32_C(  1605015226),  INT32_C(  1675849240),  INT32_C(   849752816), -INT32_C(  1643229356), -INT32_C(   513385809), -INT32_C(  2052669360) },
      UINT8_C(235),
      {  INT32_C(  2059684986), -INT32_C(   466325938), -INT32_C(  2030202088), -INT32_C(  1468571530),  INT32_C(  1442621702),  INT32_C(   285514679), -INT32_C(  1050548991),  INT32_C(    95282827) },
      { -INT32_C(  2139131598), -INT32_C(   127552545), -INT32_C(  1115790777), -INT32_C(   932776510),  INT32_C(  1444831903), -INT32_C(     9952515), -INT32_C(  1816082168), -INT32_C(   493261392) },
      { -INT32_C(  1400906683), -INT32_C(  1979184818),  INT32_C(  1605015226),  INT32_C(  1930071278),  INT32_C(   849752816), -INT32_C(  1873844529),  INT32_C(    28116855),  INT32_C(  2048363845) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_x_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_x_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_x_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r = easysimd_mm256_mask_dpwssd_epi32(src, k, a, b);
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_x_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i32x8();
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_mask_dpwssd_epi32(src, k, a, b);

    easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_maskz_dpwssd_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int32_t src[8];
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { UINT8_C(222),
      { -INT32_C(   859938024),  INT32_C(   705935048),  INT32_C(   720162868), -INT32_C(  1714834378), -INT32_C(   174710830),  INT32_C(  1375573383), -INT32_C(  1023307690),  INT32_C(  1117971241) },
      {  INT32_C(   235823174),  INT32_C(  1245192470),  INT32_C(   678700273), -INT32_C(  1413399079), -INT32_C(   459253923),  INT32_C(  1127587309),  INT32_C(  1460024878),  INT32_C(  1620682778) },
      {  INT32_C(   510633736), -INT32_C(  1167480888), -INT32_C(  1511858740),  INT32_C(  2018550555), -INT32_C(   413339142), -INT32_C(  1138060658), -INT32_C(   501993016), -INT32_C(   498881063) },
      {  INT32_C(           0),  INT32_C(   176163568),  INT32_C(   398444456),  INT32_C(  1539344745), -INT32_C(   210794716),  INT32_C(           0), -INT32_C(  1017197890),  INT32_C(  1402948926) } },
    { UINT8_C( 85),
      {  INT32_C(  1495072946),  INT32_C(  1193662313),  INT32_C(  1549978297),  INT32_C(   207084059), -INT32_C(   929415626), -INT32_C(  1701750935),  INT32_C(   410219371),  INT32_C(  1751995830) },
      { -INT32_C(  1077769386),  INT32_C(   470214498), -INT32_C(   847746894),  INT32_C(  2044317506),  INT32_C(  2000777998),  INT32_C(   890360522), -INT32_C(    61963194),  INT32_C(   811973594) },
      { -INT32_C(  1443944634), -INT32_C(  1060768242), -INT32_C(  1584579234),  INT32_C(   454714893), -INT32_C(  1533912102),  INT32_C(  1960485678),  INT32_C(    24127527),  INT32_C(   707909091) },
      {  INT32_C(  1557154900),  INT32_C(           0), -INT32_C(  2011150035),  INT32_C(           0), -INT32_C(   951396044),  INT32_C(           0),  INT32_C(    84294197),  INT32_C(           0) } },
    { UINT8_C(252),
      {  INT32_C(   386650913), -INT32_C(   713700456), -INT32_C(  1075702183), -INT32_C(  1936065232),  INT32_C(   851066511),  INT32_C(  1079651864),  INT32_C(  1948474270), -INT32_C(  1368371827) },
      { -INT32_C(  1195017440), -INT32_C(  1601291705), -INT32_C(  2124451759), -INT32_C(    49415826),  INT32_C(  1328596791), -INT32_C(  1802532107),  INT32_C(  1913172709),  INT32_C(   538998784) },
      {  INT32_C(  1004135924),  INT32_C(  1910204192),  INT32_C(  1173502679),  INT32_C(  1782775859), -INT32_C(  1111919673), -INT32_C(   497989379), -INT32_C(    78292485), -INT32_C(   988056111) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(  1223054546), -INT32_C(  1956689279),  INT32_C(    76249344),  INT32_C(   728803704),  INT32_C(  1458901965), -INT32_C(   565819155) } },
    { UINT8_C( 91),
      {  INT32_C(  1534787828),  INT32_C(   372501723), -INT32_C(   565610274),  INT32_C(   782677179), -INT32_C(  1238670483),  INT32_C(   229707444), -INT32_C(   656495517), -INT32_C(  1137466169) },
      { -INT32_C(  2145931612),  INT32_C(  2039892634),  INT32_C(  2119688131),  INT32_C(    28179859),  INT32_C(   347592800),  INT32_C(  1226926310), -INT32_C(   618528748),  INT32_C(  1217877412) },
      { -INT32_C(  1631015164), -INT32_C(  1139319047), -INT32_C(   784634050), -INT32_C(   841815956), -INT32_C(  1478391360),  INT32_C(    99615729), -INT32_C(  1478422013),  INT32_C(  1827633256) },
      { -INT32_C(  1513415812),  INT32_C(   295705887),  INT32_C(           0),  INT32_C(   780914601), -INT32_C(  1051221692),  INT32_C(           0), -INT32_C(   443491455),  INT32_C(           0) } },
    { UINT8_C( 39),
      {  INT32_C(   371264184), -INT32_C(  1856709342),  INT32_C(    16590360), -INT32_C(  2101228808), -INT32_C(  1351391060),  INT32_C(  1806858584), -INT32_C(   757900966),  INT32_C(    16400200) },
      {  INT32_C(  1796676425),  INT32_C(   301755384), -INT32_C(  1978533231),  INT32_C(  1879888580), -INT32_C(  1826652358),  INT32_C(  1392431608),  INT32_C(  1931858218),  INT32_C(  1500716816) },
      {  INT32_C(   851806778), -INT32_C(  2025602570), -INT32_C(  2146347589),  INT32_C(  1659903271), -INT32_C(  1779101539),  INT32_C(   216593377), -INT32_C(   729870908),  INT32_C(  1714287148) },
      {  INT32_C(   516979189),  INT32_C(  1856974438),  INT32_C(   974847524),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1913621120),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(125),
      { -INT32_C(  1267492622),  INT32_C(   829422300),  INT32_C(   693694220), -INT32_C(   255411488), -INT32_C(  1563337553),  INT32_C(  1365695812),  INT32_C(  1333607004),  INT32_C(  1523377000) },
      {  INT32_C(  1460551547),  INT32_C(  1166573113),  INT32_C(  1299177837),  INT32_C(  1262302619), -INT32_C(   705884271),  INT32_C(  1227248876), -INT32_C(   157768818),  INT32_C(    38888582) },
      { -INT32_C(   598122589),  INT32_C(  1260577501),  INT32_C(  1587057091),  INT32_C(  1487525574), -INT32_C(   785541147),  INT32_C(  2031768811),  INT32_C(  2121315063), -INT32_C(  1182744298) },
      { -INT32_C(  1073023711),  INT32_C(           0),  INT32_C(  1394613755),  INT32_C(    36922887), -INT32_C(  1540429811), -INT32_C(  1876122172),  INT32_C(  1722382046),  INT32_C(           0) } },
    { UINT8_C( 32),
      { -INT32_C(  1140943143),  INT32_C(  1233013176), -INT32_C(  1223696927),  INT32_C(   496789382), -INT32_C(   385323371),  INT32_C(  1004569224), -INT32_C(  1286512910), -INT32_C(  1194128418) },
      {  INT32_C(  1500762529), -INT32_C(    56430054),  INT32_C(  1437839823), -INT32_C(  1368174567),  INT32_C(  1184332734), -INT32_C(   276727811), -INT32_C(  1264397354),  INT32_C(  2137814750) },
      {  INT32_C(  1658380359), -INT32_C(  1587643694),  INT32_C(  1156976939),  INT32_C(   535980385), -INT32_C(   496596507), -INT32_C(   674044159), -INT32_C(  1735691078),  INT32_C(   840431850) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   851448919),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(216),
      {  INT32_C(  1772786927),  INT32_C(    60115954), -INT32_C(  1419388607),  INT32_C(  1435534540), -INT32_C(   766086422), -INT32_C(  1181995708), -INT32_C(  1314708039),  INT32_C(   747230524) },
      {  INT32_C(  1536504681), -INT32_C(  1084282242), -INT32_C(   798309372),  INT32_C(   858126920), -INT32_C(  1325040531),  INT32_C(  1651151273), -INT32_C(   216854858),  INT32_C(  1293917411) },
      {  INT32_C(  1302901967), -INT32_C(   485685281),  INT32_C(   347305931), -INT32_C(   548939662), -INT32_C(    24097708), -INT32_C(  1805583395), -INT32_C(   326667255), -INT32_C(   533092848) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1340678943), -INT32_C(   136238210),  INT32_C(           0), -INT32_C(  1186343644),  INT32_C(  1170809171) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i src = easysimd_x_mm256_loadu_epi32(test_vec[i].src);
    easysimd__m256i a = easysimd_x_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_x_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r = easysimd_mm256_maskz_dpwssd_epi32(k, src, a, b);
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_x_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m256i src = easysimd_test_x86_random_i32x8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_maskz_dpwssd_epi32(k, src, a, b);

    easysimd_test_x86_write_mmask8(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_dpwssd_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[16];
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { {  INT32_C(   767576910),  INT32_C(  1080917106), -INT32_C(  1575220367), -INT32_C(   477376626), -INT32_C(   355263071),  INT32_C(  2143803277),  INT32_C(  1859385624),  INT32_C(  1957233702),
        -INT32_C(  1029543601),  INT32_C(  1442975717), -INT32_C(  1460134118), -INT32_C(  1886681874),  INT32_C(   796549025),  INT32_C(  1320042806),  INT32_C(  1639809338), -INT32_C(   707434874) },
      { -INT32_C(  1265141809), -INT32_C(  1609917818), -INT32_C(  1488452935),  INT32_C(   674747526),  INT32_C(  1767354675),  INT32_C(   750257650),  INT32_C(   227374471), -INT32_C(  1428003877),
         INT32_C(  1633581786), -INT32_C(   855545836), -INT32_C(   244037014),  INT32_C(  1360636702),  INT32_C(  1320841308), -INT32_C(    59018635), -INT32_C(  1039595289),  INT32_C(  1181543531) },
      {  INT32_C(  2057816678), -INT32_C(  1656248270),  INT32_C(   277789682), -INT32_C(  1017010329), -INT32_C(  1944970217),  INT32_C(  1988726158),  INT32_C(    20484757), -INT32_C(   465066626),
        -INT32_C(  1570836881), -INT32_C(  1992317546), -INT32_C(   929444511), -INT32_C(  1953760396), -INT32_C(  1525179113), -INT32_C(  1071931350), -INT32_C(  1312730061),  INT32_C(  1754663161) },
      { -INT32_C(   259448247), -INT32_C(  2009260188), -INT32_C(  1683616957), -INT32_C(   384705201), -INT32_C(  1300653740), -INT32_C(  1848366015),  INT32_C(  1017700995),  INT32_C(  1519582622),
        -INT32_C(  1768450471),  INT32_C(  1223158748), -INT32_C(  1654185576),  INT32_C(  1814631449), -INT32_C(   400828261),  INT32_C(   615909729),  INT32_C(  2006684936), -INT32_C(   236268063) } },
    { { -INT32_C(  1945438986), -INT32_C(    82490982), -INT32_C(  1966887146), -INT32_C(  1055568214),  INT32_C(   375795180),  INT32_C(    14058189), -INT32_C(   810379306), -INT32_C(  1791539041),
        -INT32_C(   702397892), -INT32_C(  1563347061), -INT32_C(  1876126490), -INT32_C(   816758045),  INT32_C(  1021687919),  INT32_C(   272481338), -INT32_C(   203428013),  INT32_C(  1938298423) },
      { -INT32_C(   464868776), -INT32_C(   930735134), -INT32_C(  1822838096),  INT32_C(  1667410676), -INT32_C(  1667217566),  INT32_C(  1470946563),  INT32_C(    55217100), -INT32_C(    92876126),
         INT32_C(  1608433789), -INT32_C(  1943575332),  INT32_C(   186613783), -INT32_C(  1922072277), -INT32_C(   869658680), -INT32_C(  1205610772),  INT32_C(    79392098), -INT32_C(  1124191937) },
      { -INT32_C(   837034510),  INT32_C(  1482310465), -INT32_C(   295405117), -INT32_C(  1032006662), -INT32_C(   829511966), -INT32_C(   561598084),  INT32_C(  1591886110),  INT32_C(  1679483250),
        -INT32_C(    13486402),  INT32_C(  1012436089), -INT32_C(    13911035),  INT32_C(  1908582287), -INT32_C(   918597555),  INT32_C(   547931394),  INT32_C(  2021558790),  INT32_C(   702388587) },
      { -INT32_C(  1663659020), -INT32_C(   280816872),  INT32_C(  1836533738), -INT32_C(  1210858390),  INT32_C(   280911664),  INT32_C(     1748809), -INT32_C(  1285826286), -INT32_C(  1737032001),
        -INT32_C(   935307170),  INT32_C(  1509852399), -INT32_C(  1306871666), -INT32_C(   935746554),  INT32_C(  1290743789),  INT32_C(   277495050), -INT32_C(  1011789911),  INT32_C(  1423386578) } },
    { {  INT32_C(  1210650575), -INT32_C(  1585151588), -INT32_C(   861819075), -INT32_C(  1556257962), -INT32_C(  1251115853),  INT32_C(  1205212481),  INT32_C(   197088415), -INT32_C(  1137402643),
         INT32_C(  1208245676),  INT32_C(   468290014), -INT32_C(  1880651208), -INT32_C(  1590549267), -INT32_C(   514416736),  INT32_C(  1411919028),  INT32_C(  1851779201),  INT32_C(   808162180) },
      { -INT32_C(   814206991), -INT32_C(   236232008), -INT32_C(   629091604), -INT32_C(  1753500937),  INT32_C(   108646738),  INT32_C(  2136646142),  INT32_C(   250460553),  INT32_C(  1044256845),
         INT32_C(      898888),  INT32_C(    99743769), -INT32_C(  1025543733),  INT32_C(  1985567268),  INT32_C(   712823340), -INT32_C(    39200908), -INT32_C(   586443120), -INT32_C(   149206353) },
      {  INT32_C(   452405505), -INT32_C(   333453023),  INT32_C(  2125397850), -INT32_C(  2047604647),  INT32_C(  1320120794), -INT32_C(   666085304), -INT32_C(  1632217105), -INT32_C(  1567239775),
         INT32_C(   482118906), -INT32_C(   804725643),  INT32_C(   877574107), -INT32_C(  1715846209), -INT32_C(    51877452), -INT32_C(  1311492926),  INT32_C(   760187531),  INT32_C(  1456464988) },
      {  INT32_C(  1253717384), -INT32_C(  1714773303), -INT32_C(  1171231435), -INT32_C(   762406031), -INT32_C(  1566033602),  INT32_C(   329954153), -INT32_C(   304153340), -INT32_C(  1592313744),
         INT32_C(  1756508040),  INT32_C(   468019411),  INT32_C(  1666755539), -INT32_C(  1983578253), -INT32_C(   840498448),  INT32_C(  1283713936), -INT32_C(  1735365098),  INT32_C(   622274365) } },
    { { -INT32_C(   428700560),  INT32_C(  1136032616), -INT32_C(   243858382), -INT32_C(    74829497),  INT32_C(  1576497819),  INT32_C(   839830694), -INT32_C(  1302373034), -INT32_C(  1324798399),
         INT32_C(   580352954),  INT32_C(   677727734), -INT32_C(  1726358190), -INT32_C(  1449810930), -INT32_C(  1123644394), -INT32_C(  1343286184), -INT32_C(  1302245775),  INT32_C(   929327740) },
      { -INT32_C(   598082586), -INT32_C(  1694122167), -INT32_C(  1439424868), -INT32_C(   648754750), -INT32_C(  1365878186), -INT32_C(   530741905),  INT32_C(  1335082963),  INT32_C(   260503337),
         INT32_C(  1005379826),  INT32_C(  1003942303), -INT32_C(   756741361),  INT32_C(   715864532),  INT32_C(    64569748), -INT32_C(  1713162554),  INT32_C(   535328501),  INT32_C(  1613655917) },
      { -INT32_C(   291825073),  INT32_C(   455701003),  INT32_C(  1374490237), -INT32_C(   595814328), -INT32_C(  1612753447), -INT32_C(  2126986613), -INT32_C(  1499455431), -INT32_C(   569979249),
        -INT32_C(   187915543), -INT32_C(  1861225196),  INT32_C(  1289944068),  INT32_C(  1831362196),  INT32_C(  1057753011),  INT32_C(    46154953), -INT32_C(   190226331),  INT32_C(   399748910) },
      { -INT32_C(   395129779),  INT32_C(   472617048), -INT32_C(   675445630),  INT32_C(   383882375), -INT32_C(  1697328597),  INT32_C(  1596346179), -INT32_C(  1903984335), -INT32_C(  1330721460),
         INT32_C(   725814284),  INT32_C(   252973292), -INT32_C(  1956513864), -INT32_C(   786144170), -INT32_C(  1074657346), -INT32_C(  1114776050), -INT32_C(   574468798),  INT32_C(   489315068) } },
    { {  INT32_C(  1695326033), -INT32_C(  1711924331), -INT32_C(  1394222824), -INT32_C(   350614217), -INT32_C(   584440300), -INT32_C(   790566293),  INT32_C(  2026146122), -INT32_C(  1970235592),
        -INT32_C(   856712137), -INT32_C(   815405385), -INT32_C(   159626561),  INT32_C(  1809946199),  INT32_C(   659033020),  INT32_C(  1073162485), -INT32_C(   356991823), -INT32_C(  1972090797) },
      { -INT32_C(  1688837148),  INT32_C(   141278025),  INT32_C(  1560274693),  INT32_C(   969400445), -INT32_C(   513732372), -INT32_C(   366913480),  INT32_C(  1758779668),  INT32_C(    99764257),
        -INT32_C(   174044757),  INT32_C(   184355588),  INT32_C(  1869020402), -INT32_C(   911659299),  INT32_C(  1990855230),  INT32_C(  2002832226), -INT32_C(   975227740),  INT32_C(   684380540) },
      {  INT32_C(   522021658),  INT32_C(  1764301430), -INT32_C(   187134185), -INT32_C(    71466563), -INT32_C(   294492277), -INT32_C(   681192909), -INT32_C(  2086845433),  INT32_C(   816539413),
         INT32_C(  1229965522), -INT32_C(    88966941), -INT32_C(   974222841), -INT32_C(  1748980980),  INT32_C(  1166357010),  INT32_C(   203287045),  INT32_C(  1150269998), -INT32_C(   193709279) },
      { -INT32_C(  2103766217), -INT32_C(  1773071010), -INT32_C(  1277368141), -INT32_C(   106001557), -INT32_C(   434383510), -INT32_C(   997162088),  INT32_C(   998084314), -INT32_C(  1464000109),
        -INT32_C(  1172959523), -INT32_C(   731954899), -INT32_C(   559957967),  INT32_C(  1920292211),  INT32_C(  1233279082),  INT32_C(  1243724735), -INT32_C(   859038278),  INT32_C(  2111758551) } },
    { { -INT32_C(   432159997),  INT32_C(  1122037563), -INT32_C(  2063085959), -INT32_C(  1944205191), -INT32_C(     3038470), -INT32_C(  1156845939), -INT32_C(   889218136), -INT32_C(   641829930),
         INT32_C(  1908472630),  INT32_C(  1672782058), -INT32_C(   387335313),  INT32_C(  2121533059),  INT32_C(   897402536), -INT32_C(   571373260), -INT32_C(    89657308), -INT32_C(  1697421980) },
      {  INT32_C(  1242338144), -INT32_C(  1548894156), -INT32_C(     7629189),  INT32_C(  1165820060),  INT32_C(  2054879814), -INT32_C(  1487443069), -INT32_C(  1063125412), -INT32_C(  1017481885),
         INT32_C(  1007511048), -INT32_C(  1562395866), -INT32_C(   291411119), -INT32_C(  1322049941), -INT32_C(  1674859240),  INT32_C(  1967358745), -INT32_C(   449452671),  INT32_C(  1655213914) },
      {  INT32_C(   480229110), -INT32_C(  1027703183),  INT32_C(  1420845033), -INT32_C(  1777998979), -INT32_C(  1439551087),  INT32_C(   891254196), -INT32_C(  1273342886), -INT32_C(   652753949),
        -INT32_C(   352930183),  INT32_C(   514634804), -INT32_C(  1854776044), -INT32_C(   785942464),  INT32_C(  1568364969),  INT32_C(   697473743), -INT32_C(   757158418), -INT32_C(   374606480) },
      {  INT32_C(   226684599),  INT32_C(   964031125),  INT32_C(  1566094588),  INT32_C(  1866990102), -INT32_C(   710362508), -INT32_C(   636179406), -INT32_C(   583074726), -INT32_C(   949630283),
         INT32_C(  1334016892),  INT32_C(  1827597430),  INT32_C(   393150325), -INT32_C(  1691784440), -INT32_C(   201356503),  INT32_C(   576401313),  INT32_C(   132102408), -INT32_C(  1763832580) } },
    { { -INT32_C(   539778645),  INT32_C(  1794998102),  INT32_C(   503017692), -INT32_C(  1846664216),  INT32_C(  1273916028), -INT32_C(   210403324), -INT32_C(  1648012499), -INT32_C(   209293240),
         INT32_C(  1758615826), -INT32_C(  1244475175),  INT32_C(   684903744),  INT32_C(  1840890352),  INT32_C(   800630571),  INT32_C(  1428303143), -INT32_C(   923605120),  INT32_C(  1790671192) },
      { -INT32_C(  1412264238), -INT32_C(  1637768098),  INT32_C(  1657156465),  INT32_C(   533692404),  INT32_C(  1297057574),  INT32_C(   899838389), -INT32_C(  1308715687), -INT32_C(   535054066),
        -INT32_C(  1517490873),  INT32_C(    38006161),  INT32_C(   358877472), -INT32_C(  1372310648),  INT32_C(  1895596987),  INT32_C(  1319476981),  INT32_C(  1107272499), -INT32_C(  1541268899) },
      { -INT32_C(  1739936249), -INT32_C(  1147499109),  INT32_C(   533790615), -INT32_C(   305265358),  INT32_C(  2103298696), -INT32_C(  1664416920),  INT32_C(    64866982), -INT32_C(   341245980),
         INT32_C(  1216606893),  INT32_C(   369368703),  INT32_C(  1328993309),  INT32_C(  1648166105),  INT32_C(   920623822),  INT32_C(  1154657181),  INT32_C(  1514647669),  INT32_C(  1564864431) },
      {  INT32_C(   640393245), -INT32_C(  1375409254),  INT32_C(   707564131), -INT32_C(  1841028830), -INT32_C(  1962419361), -INT32_C(   533721086), -INT32_C(  1294858455), -INT32_C(   166261384),
         INT32_C(  1345119921), -INT32_C(  1278053164),  INT32_C(   769701112),  INT32_C(  1330641564),  INT32_C(  2031073577), -INT32_C(  1975277630), -INT32_C(    49380984),  INT32_C(  1200941333) } },
    { {  INT32_C(  1638255073),  INT32_C(    74951143),  INT32_C(  1465101694),  INT32_C(  2142867633), -INT32_C(   927557333), -INT32_C(  1190361020), -INT32_C(   401386440),  INT32_C(   625301827),
         INT32_C(   159836706),  INT32_C(   302906772),  INT32_C(  1550409899),  INT32_C(   484189169),  INT32_C(    14979772),  INT32_C(  1404694810), -INT32_C(  2009346747),  INT32_C(  1219330086) },
      { -INT32_C(    28232854), -INT32_C(   602907087), -INT32_C(  1355253058),  INT32_C(  1506481309), -INT32_C(  1051086682), -INT32_C(   434892127),  INT32_C(    91115487),  INT32_C(   961354959),
        -INT32_C(  2143773105), -INT32_C(  1151579908),  INT32_C(  1600820674),  INT32_C(  1354249897), -INT32_C(  2028924442),  INT32_C(    57484580),  INT32_C(  1124719476),  INT32_C(  1199331063) },
      { -INT32_C(   238570251), -INT32_C(  1079237379),  INT32_C(  1646204857),  INT32_C(   867358541),  INT32_C(   230343657),  INT32_C(  1544562664), -INT32_C(    90236670),  INT32_C(  1698765936),
        -INT32_C(   849999664), -INT32_C(   443743444),  INT32_C(  1732815898),  INT32_C(  1822096003), -INT32_C(  1501997890),  INT32_C(  2097384059),  INT32_C(   326607523), -INT32_C(  1887913794) },
      {  INT32_C(  1386969162),  INT32_C(   454782740),  INT32_C(  1136466580), -INT32_C(  1902870768), -INT32_C(   671428117), -INT32_C(  1298038164), -INT32_C(   272398776),  INT32_C(  1059226976),
         INT32_C(   527691042),  INT32_C(   437058192), -INT32_C(  1514575313),  INT32_C(  1039029532),  INT32_C(   823931001),  INT32_C(  1146713261), -INT32_C(  1700215728),  INT32_C(   286022040) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi32(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r = easysimd_mm512_dpwssd_epi32(src, a, b);
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i32x16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    easysimd__m512i r = easysimd_mm512_dpwssd_epi32(src, a, b);

    easysimd_test_x86_write_i32x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_dpwssd_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[16];
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(   312684607), -INT32_C(   321656366),  INT32_C(   424876949), -INT32_C(   746197739), -INT32_C(  1132789951),  INT32_C(   741965193), -INT32_C(   566185697),  INT32_C(   711833705),
         INT32_C(  1494796679),  INT32_C(  1212541875),  INT32_C(   459380742), -INT32_C(   957356155),  INT32_C(  1870817766),  INT32_C(    94092518), -INT32_C(   689710227),  INT32_C(   453070996) },
      UINT16_C( 6426),
      { -INT32_C(  1174090379),  INT32_C(  2001930773),  INT32_C(   341760037),  INT32_C(   561857695),  INT32_C(  1323131827),  INT32_C(  1277839977),  INT32_C(   580763169),  INT32_C(  1312536537),
        -INT32_C(  1727447164),  INT32_C(  1846565961), -INT32_C(   746361292),  INT32_C(  1710555314), -INT32_C(   843787676),  INT32_C(  1041882653),  INT32_C(  1969272476), -INT32_C(   238773395) },
      {  INT32_C(   596364762),  INT32_C(  1569823529), -INT32_C(  1137634294),  INT32_C(  2032281109),  INT32_C(   356964088),  INT32_C(  1347641268), -INT32_C(  2084195562),  INT32_C(   678791502),
         INT32_C(  2135621718), -INT32_C(  1512186469),  INT32_C(   107089649),  INT32_C(   746554420),  INT32_C(   239256922),  INT32_C(   996054309), -INT32_C(  1749146807),  INT32_C(    62927789) },
      { -INT32_C(   312684607),  INT32_C(   343407410),  INT32_C(   424876949), -INT32_C(   309088326), -INT32_C(  1290934249),  INT32_C(   741965193), -INT32_C(   566185697),  INT32_C(   711833705),
         INT32_C(   637234306),  INT32_C(  1212541875),  INT32_C(   459380742), -INT32_C(   665680840),  INT32_C(  1999123830),  INT32_C(    94092518), -INT32_C(   689710227),  INT32_C(   453070996) } },
    { { -INT32_C(   813495501), -INT32_C(   629907224), -INT32_C(  1562323346), -INT32_C(  1261543334),  INT32_C(  1254232101), -INT32_C(   276488026), -INT32_C(   242858940), -INT32_C(  1410054537),
         INT32_C(   981104466),  INT32_C(  1158999767),  INT32_C(   535295429),  INT32_C(  2060760661),  INT32_C(  1824888518), -INT32_C(    77903177),  INT32_C(   116187790),  INT32_C(  2058477608) },
      UINT16_C(11096),
      { -INT32_C(   921096267),  INT32_C(  1556012661), -INT32_C(   770567170),  INT32_C(  1399380366), -INT32_C(  1583538363),  INT32_C(   109259802), -INT32_C(   471356622), -INT32_C(   619823322),
        -INT32_C(   475781266),  INT32_C(    71328518),  INT32_C(    81154678),  INT32_C(  1901542955),  INT32_C(  2014508382),  INT32_C(  1400870177),  INT32_C(  1714841152),  INT32_C(   323044517) },
      {  INT32_C(  1912071787), -INT32_C(  1082837175), -INT32_C(  1245426807), -INT32_C(   400155766),  INT32_C(   828389392),  INT32_C(   226811853), -INT32_C(   361448891),  INT32_C(  1795077630),
        -INT32_C(   455346789), -INT32_C(  1247522516),  INT32_C(   644507804), -INT32_C(  1810984829), -INT32_C(  1782222904), -INT32_C(  1801237938),  INT32_C(    58595076),  INT32_C(  1752005836) },
      { -INT32_C(   813495501), -INT32_C(   629907224), -INT32_C(  1562323346), -INT32_C(  1461304746),  INT32_C(  1063699541), -INT32_C(   276488026),  INT32_C(   184362858), -INT32_C(  1410054537),
         INT32_C(  1004694072),  INT32_C(  1665061599),  INT32_C(   535295429),  INT32_C(   804730136),  INT32_C(  1824888518), -INT32_C(  1185817950),  INT32_C(   116187790),  INT32_C(  2058477608) } },
    { { -INT32_C(  1655945103),  INT32_C(   911470745), -INT32_C(   597901992),  INT32_C(   376466254),  INT32_C(   682374618), -INT32_C(  2068033665),  INT32_C(   847723366),  INT32_C(   698021047),
        -INT32_C(   691607748),  INT32_C(   806099415),  INT32_C(   621570263), -INT32_C(  1388610349),  INT32_C(   836167601), -INT32_C(  1665822154), -INT32_C(  2050016051),  INT32_C(  1823369520) },
      UINT16_C(29776),
      {  INT32_C(  1317939266),  INT32_C(  1689740632), -INT32_C(   975140214),  INT32_C(   246255928),  INT32_C(  2007032770),  INT32_C(  1337159296), -INT32_C(  1581718541), -INT32_C(  1844115120),
        -INT32_C(  1981701327), -INT32_C(  1829922808),  INT32_C(  1515769122),  INT32_C(   560465246),  INT32_C(  1754794472),  INT32_C(  1790397559),  INT32_C(  2131456047), -INT32_C(  1441652615) },
      { -INT32_C(   869010492), -INT32_C(  1386274677),  INT32_C(  1258862573), -INT32_C(  1536397124), -INT32_C(   267582087), -INT32_C(  2141535151), -INT32_C(  1392483020),  INT32_C(  1247220358),
        -INT32_C(  1877571067), -INT32_C(  1757579862), -INT32_C(   387758804),  INT32_C(   797790134), -INT32_C(  1524589996), -INT32_C(  1826194850),  INT32_C(  1715480288),  INT32_C(  1034983480) },
      { -INT32_C(  1655945103),  INT32_C(   911470745), -INT32_C(   597901992),  INT32_C(   376466254),  INT32_C(   547207596), -INT32_C(  2068033665),  INT32_C(  1174519746),  INT32_C(   698021047),
        -INT32_C(   691607748),  INT32_C(   806099415),  INT32_C(   254128311), -INT32_C(  1388610349),  INT32_C(   147243729), -INT32_C(  1813886958), -INT32_C(   912882515),  INT32_C(  1823369520) } },
    { { -INT32_C(   842152414),  INT32_C(  1751386684),  INT32_C(   105989968), -INT32_C(   348791145), -INT32_C(   695118472), -INT32_C(  1335248944),  INT32_C(   353807069),  INT32_C(  1683146306),
        -INT32_C(   936304756),  INT32_C(  2033292841),  INT32_C(  1954513629), -INT32_C(   664750752), -INT32_C(   626069238), -INT32_C(  2054547288),  INT32_C(    77242562), -INT32_C(   211227546) },
      UINT16_C(39435),
      { -INT32_C(   332385093),  INT32_C(   762187182), -INT32_C(   488452478),  INT32_C(  1439952294),  INT32_C(  1382906823), -INT32_C(  1695404288), -INT32_C(  1685694157),  INT32_C(   120951116),
         INT32_C(  1945396677), -INT32_C(   173972877), -INT32_C(   673742031),  INT32_C(   942451312),  INT32_C(   629840165), -INT32_C(    71336760),  INT32_C(   563561941), -INT32_C(  1691759402) },
      { -INT32_C(  1525736142), -INT32_C(  1332039553), -INT32_C(  1534561997),  INT32_C(  1088205851),  INT32_C(   375744078), -INT32_C(  1206770462),  INT32_C(  1071229289), -INT32_C(  1478884747),
        -INT32_C(  1622349537), -INT32_C(   850401639),  INT32_C(  1920063319), -INT32_C(   625848948), -INT32_C(  1779427405), -INT32_C(  1538457029),  INT32_C(   568534955),  INT32_C(  1237892649) },
      { -INT32_C(   623179656),  INT32_C(  1443719098),  INT32_C(   105989968),  INT32_C(   107292781), -INT32_C(   695118472), -INT32_C(  1335248944),  INT32_C(   353807069),  INT32_C(  1683146306),
        -INT32_C(   936304756),  INT32_C(  1903851667),  INT32_C(  1954513629), -INT32_C(  1236917960), -INT32_C(  1046750135), -INT32_C(  2054547288),  INT32_C(    77242562), -INT32_C(   478054476) } },
    { {  INT32_C(  1105728935),  INT32_C(  1393440763), -INT32_C(  1681555697),  INT32_C(  2138405068), -INT32_C(   887855729),  INT32_C(   326066792), -INT32_C(  1305193591), -INT32_C(  1191445231),
         INT32_C(   234480402),  INT32_C(   710936347),  INT32_C(  1388652166),  INT32_C(   768686750),  INT32_C(   133752479), -INT32_C(   786732984), -INT32_C(   863809605),  INT32_C(  1585741644) },
      UINT16_C(32354),
      { -INT32_C(   863666836),  INT32_C(  1827802279),  INT32_C(   816222302), -INT32_C(  1240054082),  INT32_C(  1746820685), -INT32_C(  1296508625),  INT32_C(   707855525), -INT32_C(   811035549),
        -INT32_C(  1181012719), -INT32_C(  1742369223), -INT32_C(   590820322),  INT32_C(  1603526162),  INT32_C(  1808314684),  INT32_C(   807239819), -INT32_C(   413511804), -INT32_C(   189398301) },
      {  INT32_C(  1772966448), -INT32_C(    33434656), -INT32_C(  1310996065), -INT32_C(   485462361), -INT32_C(  1437673441), -INT32_C(   572822440), -INT32_C(  1631308357),  INT32_C(  1754430264),
        -INT32_C(  1378795315), -INT32_C(  1297362157),  INT32_C(  1147373212),  INT32_C(   338129908), -INT32_C(  1531021748), -INT32_C(  1652450846),  INT32_C(   121325263), -INT32_C(  1905275455) },
      {  INT32_C(  1105728935),  INT32_C(  1342601229), -INT32_C(  1681555697),  INT32_C(  2138405068), -INT32_C(   887855729),  INT32_C(   223386104), -INT32_C(  1557697388), -INT32_C(  1191445231),
         INT32_C(   234480402),  INT32_C(  1572268813),  INT32_C(  1643560486),  INT32_C(   637076779), -INT32_C(  1121498401), -INT32_C(   244318821), -INT32_C(   506496203),  INT32_C(  1585741644) } },
    { {  INT32_C(   540753933), -INT32_C(  1328355821),  INT32_C(  1643394413), -INT32_C(   176874583),  INT32_C(  1939420305), -INT32_C(  1676665907),  INT32_C(   581127009),  INT32_C(   649073177),
         INT32_C(  1715989331),  INT32_C(  1041635793), -INT32_C(   123729329), -INT32_C(  1209199322),  INT32_C(   388728393),  INT32_C(    45300641), -INT32_C(  1608231033), -INT32_C(  1127820183) },
      UINT16_C( 3519),
      {  INT32_C(   958894371),  INT32_C(  1866757839),  INT32_C(  1535470190),  INT32_C(  1289866785), -INT32_C(  1719106587), -INT32_C(  1427107963),  INT32_C(  1971214767),  INT32_C(   964902422),
        -INT32_C(  1636652337), -INT32_C(  1911638496),  INT32_C(  1122603808),  INT32_C(  1200540257), -INT32_C(   740288947), -INT32_C(   729952219),  INT32_C(  1061878569),  INT32_C(   142134585) },
      { -INT32_C(  1750602889), -INT32_C(  1037716063), -INT32_C(  1459351992),  INT32_C(   653300440), -INT32_C(   822488919), -INT32_C(   895256928), -INT32_C(  1425412751),  INT32_C(   833847994),
         INT32_C(   248011373),  INT32_C(  1473309967), -INT32_C(   754854661),  INT32_C(   284815719),  INT32_C(  1658778306), -INT32_C(   634617496),  INT32_C(   679818862),  INT32_C(   626604216) },
      {  INT32_C(   299114947),  INT32_C(  1936496150),  INT32_C(  1221288817),  INT32_C(   376066673), -INT32_C(  1641276410), -INT32_C(  1258531907),  INT32_C(   581127009),  INT32_C(   326673326),
         INT32_C(  1116631174),  INT32_C(  1041635793), -INT32_C(    14083944), -INT32_C(  1078309813),  INT32_C(   388728393),  INT32_C(    45300641), -INT32_C(  1608231033), -INT32_C(  1127820183) } },
    { { -INT32_C(  1573707373),  INT32_C(   167379982),  INT32_C(  1071512536), -INT32_C(  1370499348),  INT32_C(   823209673),  INT32_C(   504053167),  INT32_C(   726044787),  INT32_C(  1548787913),
        -INT32_C(   788626239),  INT32_C(  1624897672), -INT32_C(   526338317),  INT32_C(  1452208013), -INT32_C(   846749923),  INT32_C(  1357616093), -INT32_C(   327405277), -INT32_C(  1823945519) },
      UINT16_C(18256),
      {  INT32_C(  1010817123), -INT32_C(   638373063), -INT32_C(  1580695533),  INT32_C(  1564600022), -INT32_C(  1628430669), -INT32_C(   355527570),  INT32_C(  1236771072), -INT32_C(  1752168652),
         INT32_C(   433311712),  INT32_C(   368231938),  INT32_C(   481802822),  INT32_C(  1400567968), -INT32_C(  2064553450),  INT32_C(  2138030462), -INT32_C(  1765267870),  INT32_C(   221075501) },
      {  INT32_C(   707133479),  INT32_C(   222238919),  INT32_C(  1932195539),  INT32_C(    80192750), -INT32_C(  1920354290), -INT32_C(   619906951),  INT32_C(  1265751070),  INT32_C(  1398382124),
         INT32_C(  1719500703),  INT32_C(  1802747288), -INT32_C(  1562468941),  INT32_C(  1353098561), -INT32_C(   673370274),  INT32_C(  1169418535), -INT32_C(   376363843),  INT32_C(  1648159427) },
      { -INT32_C(  1573707373),  INT32_C(   167379982),  INT32_C(  1071512536), -INT32_C(  1370499348),  INT32_C(  1406583555),  INT32_C(   504053167),  INT32_C(  1363711290),  INT32_C(  1548787913),
        -INT32_C(  1017680152),  INT32_C(  2031813998), -INT32_C(   252405609),  INT32_C(  1452208013), -INT32_C(   846749923),  INT32_C(  1357616093), -INT32_C(    80298299), -INT32_C(  1823945519) } },
    { {  INT32_C(    29932137),  INT32_C(   711736183),  INT32_C(   449596377),  INT32_C(  1315599344), -INT32_C(   903460958), -INT32_C(   317728464), -INT32_C(  1059675907), -INT32_C(   199093366),
         INT32_C(  1156967117), -INT32_C(     9543130), -INT32_C(  1659290707),  INT32_C(  1357677742), -INT32_C(    65400117), -INT32_C(   387372309),  INT32_C(  1437188298), -INT32_C(  1605776429) },
      UINT16_C(16054),
      {  INT32_C(  1386274020), -INT32_C(   192066085), -INT32_C(   679986197), -INT32_C(  1494662004),  INT32_C(   701551680),  INT32_C(  1709808572),  INT32_C(   959560944),  INT32_C(  1081599836),
        -INT32_C(  1634527037),  INT32_C(  1351753829), -INT32_C(   400094372), -INT32_C(  1886515121), -INT32_C(  1581752348), -INT32_C(   368663814), -INT32_C(  1155319970), -INT32_C(   503604706) },
      {  INT32_C(   411012786),  INT32_C(   191369903),  INT32_C(  1827901469), -INT32_C(  2047114848), -INT32_C(   634997792), -INT32_C(  1262212011), -INT32_C(  2106595484),  INT32_C(   878996098),
        -INT32_C(  1471355911),  INT32_C(   313767413), -INT32_C(   444684731),  INT32_C(   141195559), -INT32_C(  2099081171),  INT32_C(   557230013),  INT32_C(   279160206),  INT32_C(   155518736) },
      {  INT32_C(    29932137),  INT32_C(   798507636), -INT32_C(   279155616),  INT32_C(  1315599344), -INT32_C(   788917118), -INT32_C(  1111508736), -INT32_C(  1059675907), -INT32_C(   149749058),
         INT32_C(  1156967117), -INT32_C(    68000283), -INT32_C(  1684661845),  INT32_C(  1425160323),  INT32_C(     4940367),  INT32_C(   102872769),  INT32_C(  1437188298), -INT32_C(  1605776429) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi32(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r = easysimd_mm512_mask_dpwssd_epi32(src, test_vec[i].k, a, b);
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i32x16();
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    easysimd__m512i r = easysimd_mm512_mask_dpwssd_epi32(src, k, a, b);

    easysimd_test_x86_write_i32x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_maskz_dpwssd_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k;
    const int32_t src[16];
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { UINT16_C(37355),
      {  INT32_C(  1682366641),  INT32_C(  1896516595), -INT32_C(   605343375),  INT32_C(   476846137), -INT32_C(   775738982), -INT32_C(   311013046), -INT32_C(  1477081502),  INT32_C(  1077469327),
        -INT32_C(  1281065024),  INT32_C(  2066066954),  INT32_C(   425136352), -INT32_C(  1036664024),  INT32_C(   915667180), -INT32_C(  1406989750),  INT32_C(   542316688), -INT32_C(  1184855048) },
      {  INT32_C(   359400714), -INT32_C(  1819242061), -INT32_C(   911349855), -INT32_C(  1769151830),  INT32_C(   634134491), -INT32_C(  1160646615),  INT32_C(    31073288), -INT32_C(  1162200401),
        -INT32_C(   204528065),  INT32_C(  1501978552), -INT32_C(   249351353), -INT32_C(   242700522), -INT32_C(   132754226),  INT32_C(  1303570244), -INT32_C(  1169257461),  INT32_C(   108267718) },
      { -INT32_C(   419871954), -INT32_C(   364871773), -INT32_C(   908368973), -INT32_C(   524655854), -INT32_C(    52899656), -INT32_C(  1035367753), -INT32_C(   579037418), -INT32_C(   823922273),
        -INT32_C(   676013004),  INT32_C(   247592027),  INT32_C(  1775738199), -INT32_C(  1203072768),  INT32_C(   414524001), -INT32_C(  1009058131),  INT32_C(   899700630),  INT32_C(  2080605000) },
      {  INT32_C(  1669416073),  INT32_C(  1128440300),  INT32_C(           0),  INT32_C(   502507493),  INT32_C(           0),  INT32_C(    90554418), -INT32_C(  1729007958),  INT32_C(  1245139246),
        -INT32_C(  1338593512),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1093922440),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1204119292) } },
    { UINT16_C(47199),
      {  INT32_C(   363641684), -INT32_C(  1582168887), -INT32_C(  1221414035),  INT32_C(   551195755), -INT32_C(  2027976788), -INT32_C(   337726134), -INT32_C(   294770966), -INT32_C(   156840286),
         INT32_C(  1376539273), -INT32_C(   990658986), -INT32_C(   579131791),  INT32_C(  1727878586),  INT32_C(   669850844), -INT32_C(  1139618863), -INT32_C(  1800765199), -INT32_C(   678801330) },
      { -INT32_C(   114649438), -INT32_C(   960684715), -INT32_C(    39634877),  INT32_C(  1784979854), -INT32_C(  1886301763),  INT32_C(   239837981),  INT32_C(  1923282212), -INT32_C(   414569147),
         INT32_C(   417363139), -INT32_C(   706765423),  INT32_C(  1691517654), -INT32_C(   506579421), -INT32_C(  1519361912),  INT32_C(   666090243), -INT32_C(   174434640),  INT32_C(  1188881539) },
      { -INT32_C(   379601576),  INT32_C(   817774170), -INT32_C(   460025408),  INT32_C(  1338336199), -INT32_C(   957073981), -INT32_C(  1594971920), -INT32_C(  2120906498), -INT32_C(   993562260),
        -INT32_C(  1985141202),  INT32_C(   632908900), -INT32_C(  1006022915),  INT32_C(  1947520689), -INT32_C(   214300669), -INT32_C(  1366087504),  INT32_C(   472852656), -INT32_C(   924780647) },
      {  INT32_C(   834063770), -INT32_C(  1645225231), -INT32_C(  1623677207),  INT32_C(   489803457), -INT32_C(  1319639081),  INT32_C(           0), -INT32_C(  1159800084),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1323231749),  INT32_C(   796346836), -INT32_C(  1535423574),  INT32_C(           0), -INT32_C(   919656423) } },
    { UINT16_C(36381),
      {  INT32_C(   184189521), -INT32_C(  1336346713), -INT32_C(   813823557), -INT32_C(  1193770627),  INT32_C(   132155508),  INT32_C(  1714393399), -INT32_C(  1906456403), -INT32_C(   518227056),
        -INT32_C(  1528031491), -INT32_C(   934001651), -INT32_C(   879177138), -INT32_C(   947687341),  INT32_C(   802055416), -INT32_C(  1567162891),  INT32_C(  1446048709),  INT32_C(  1798786158) },
      {  INT32_C(  1863263074), -INT32_C(  1237818520), -INT32_C(  2004758475),  INT32_C(   944702784),  INT32_C(  1583881577), -INT32_C(   536805862),  INT32_C(  1614164465), -INT32_C(   523539075),
        -INT32_C(   112206959),  INT32_C(  1957660478), -INT32_C(  1745080233), -INT32_C(  1630516171), -INT32_C(  2063845271),  INT32_C(   660929846), -INT32_C(  1417176530), -INT32_C(  1735699705) },
      {  INT32_C(  1821498158), -INT32_C(  1159708574), -INT32_C(  1504584335), -INT32_C(  1840963287), -INT32_C(  1894366887),  INT32_C(  1823963710),  INT32_C(   471350804), -INT32_C(  1078680687),
        -INT32_C(   517192066), -INT32_C(   140833658),  INT32_C(   312405225),  INT32_C(  1738924814),  INT32_C(  1677179684),  INT32_C(  1255124534),  INT32_C(  2120673260),  INT32_C(   154999691) },
      {  INT32_C(   888991596),  INT32_C(           0), -INT32_C(     3241743), -INT32_C(  1587293176), -INT32_C(   440492059),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0), -INT32_C(  1097304714), -INT32_C(  1066562811), -INT32_C(  1752390071),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1886018138) } },
    { UINT16_C(26977),
      { -INT32_C(  2055804950),  INT32_C(  2104647647),  INT32_C(   392200306), -INT32_C(   556628761), -INT32_C(  1215559449), -INT32_C(  1180796589),  INT32_C(   886319607),  INT32_C(   496907571),
        -INT32_C(    56421347), -INT32_C(   445049485),  INT32_C(  2096945557),  INT32_C(  1163644765),  INT32_C(   721217495),  INT32_C(  1474599520), -INT32_C(   141772604),  INT32_C(   185871086) },
      { -INT32_C(  1324894402),  INT32_C(  1637318860), -INT32_C(  1226927271),  INT32_C(   989542754), -INT32_C(  2124089568),  INT32_C(  1457014930), -INT32_C(   280140799), -INT32_C(   856006258),
        -INT32_C(   444727015), -INT32_C(   632875647),  INT32_C(   193996200),  INT32_C(  2118487134),  INT32_C(   369076611), -INT32_C(   211035918), -INT32_C(   874334148),  INT32_C(   865590298) },
      {  INT32_C(  1578636765), -INT32_C(   751280342), -INT32_C(   488716156), -INT32_C(   664722604), -INT32_C(  1091739444),  INT32_C(  1974556984),  INT32_C(   725652241),  INT32_C(  1281283951),
         INT32_C(   397047789),  INT32_C(  1542120150), -INT32_C(    12728149), -INT32_C(  1193828629),  INT32_C(   930530815),  INT32_C(   799811614),  INT32_C(   727444667), -INT32_C(  1317553724) },
      {  INT32_C(  1647925528),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(    86647141),  INT32_C(   125051720),  INT32_C(           0),
        -INT32_C(    88903954),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1310478746),  INT32_C(           0),  INT32_C(  1332610976), -INT32_C(   199226386),  INT32_C(           0) } },
    { UINT16_C( 8752),
      { -INT32_C(  1291581496), -INT32_C(  1619284126), -INT32_C(  2042730577), -INT32_C(  1790231265),  INT32_C(   566061428), -INT32_C(   200443495),  INT32_C(   464376228),  INT32_C(  1262345858),
         INT32_C(  1207845605), -INT32_C(  1595508239), -INT32_C(    14211872), -INT32_C(   694848927),  INT32_C(  1979142876),  INT32_C(  1869153483),  INT32_C(  1485510358), -INT32_C(   643577612) },
      { -INT32_C(    64970486), -INT32_C(    73660645), -INT32_C(  1912880341),  INT32_C(   308514870), -INT32_C(  1366861086),  INT32_C(   874377310), -INT32_C(    74602490),  INT32_C(  2077503601),
        -INT32_C(   310905390),  INT32_C(   669521148),  INT32_C(   229958615),  INT32_C(  1444878195), -INT32_C(   805001615), -INT32_C(  1660673642),  INT32_C(   999854538), -INT32_C(  1799983934) },
      {  INT32_C(  1568747105),  INT32_C(   428173634), -INT32_C(  1071171251), -INT32_C(  1038727599), -INT32_C(  2087576851),  INT32_C(   102799164), -INT32_C(   364791768), -INT32_C(  2038499547),
         INT32_C(  1743060773), -INT32_C(  1233032855), -INT32_C(   210327390), -INT32_C(   608793106), -INT32_C(   480294745),  INT32_C(   149520351),  INT32_C(  1576151608),  INT32_C(  1206087970) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1390812844), -INT32_C(    72191167),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0), -INT32_C(  1642758859),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  2092808473),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(51056),
      {  INT32_C(   808507823),  INT32_C(   114873231),  INT32_C(  2090059462),  INT32_C(    29702818),  INT32_C(   125871134), -INT32_C(  1624065876), -INT32_C(   116370411),  INT32_C(  1254129819),
        -INT32_C(   377818790), -INT32_C(  1964027196), -INT32_C(  1157135847), -INT32_C(   608383811),  INT32_C(   501365872),  INT32_C(   180098293),  INT32_C(    67357800), -INT32_C(  1504787380) },
      {  INT32_C(  2039531957),  INT32_C(   872644379), -INT32_C(  1074853374),  INT32_C(  1201318870), -INT32_C(   580616984), -INT32_C(   102293359),  INT32_C(   956165100),  INT32_C(  1692355759),
         INT32_C(   819818261), -INT32_C(   261758738), -INT32_C(  1045474069), -INT32_C(   402110208),  INT32_C(  1472556230),  INT32_C(  2018552972),  INT32_C(  1186025111), -INT32_C(  1347776358) },
      { -INT32_C(   304117761),  INT32_C(  1407075432), -INT32_C(  1726706279), -INT32_C(  1652482601),  INT32_C(   385107593), -INT32_C(  1970387726),  INT32_C(   751845266), -INT32_C(   824476721),
         INT32_C(  1790753538), -INT32_C(  1732404481), -INT32_C(    30289369),  INT32_C(  2023469807), -INT32_C(   342978568),  INT32_C(  1718951380),  INT32_C(   747849052), -INT32_C(  1006997824) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   651200406), -INT32_C(  1429984800), -INT32_C(    32653763),  INT32_C(           0),
        -INT32_C(   538277168), -INT32_C(  1652752793), -INT32_C(  1404900251),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   631095631), -INT32_C(  1997312560) } },
    { UINT16_C(46633),
      { -INT32_C(   347068115), -INT32_C(   205686591),  INT32_C(   312847478), -INT32_C(  1281253852), -INT32_C(    19892855),  INT32_C(  1866673372),  INT32_C(  1406993496), -INT32_C(   200734777),
        -INT32_C(   237020624),  INT32_C(  1222941906),  INT32_C(  1834650184), -INT32_C(  1323238360),  INT32_C(  1303375985),  INT32_C(  1975317020), -INT32_C(  1110926602), -INT32_C(   776810079) },
      { -INT32_C(    37580501),  INT32_C(  1984341806),  INT32_C(  1508089905),  INT32_C(   235537308),  INT32_C(   274446835), -INT32_C(  1568335701),  INT32_C(  1398754738),  INT32_C(  1243877662),
        -INT32_C(   783817053), -INT32_C(  1102606963), -INT32_C(   904451282),  INT32_C(   584589614), -INT32_C(  2043530022), -INT32_C(    30886068),  INT32_C(   575768324),  INT32_C(  1013740953) },
      { -INT32_C(   384977829),  INT32_C(  1873237057), -INT32_C(  1371881601), -INT32_C(  1143991584), -INT32_C(  1841233338), -INT32_C(  1114609223), -INT32_C(  1981816336), -INT32_C(  1295692714),
         INT32_C(  1100730880), -INT32_C(  1498332633), -INT32_C(   497751295),  INT32_C(  1151149309), -INT32_C(   539566554),  INT32_C(   932996935), -INT32_C(  1614709688), -INT32_C(   934181176) },
      {  INT32_C(   205747280),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1339525948),  INT32_C(           0), -INT32_C(  1850360001),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0),  INT32_C(  1111433316),  INT32_C(  1880067330),  INT32_C(           0),  INT32_C(  1442877593),  INT32_C(  1476514960),  INT32_C(           0), -INT32_C(  1931524779) } },
    { UINT16_C(60505),
      { -INT32_C(  1171357687),  INT32_C(  2074423334), -INT32_C(  1348492526), -INT32_C(  1114781977),  INT32_C(  1109709989), -INT32_C(   859935477),  INT32_C(  1565754892),  INT32_C(  1464446030),
         INT32_C(  1376941868), -INT32_C(  1177700441),  INT32_C(  1097362778), -INT32_C(   654314189), -INT32_C(   719707191), -INT32_C(  1667114864), -INT32_C(  1392905122), -INT32_C(   872136032) },
      {  INT32_C(  1629427386),  INT32_C(   672918733), -INT32_C(  1939242151),  INT32_C(  1113942137),  INT32_C(   471302028), -INT32_C(  1229342376),  INT32_C(  1315156653), -INT32_C(  1357223947),
         INT32_C(  1242577277),  INT32_C(  2138188582),  INT32_C(   671931567), -INT32_C(   798330556),  INT32_C(  1223525104), -INT32_C(   385898949),  INT32_C(  1295409752),  INT32_C(  1190941129) },
      { -INT32_C(  1332736630), -INT32_C(   416283848),  INT32_C(   588201183), -INT32_C(  1644922195),  INT32_C(   937878012), -INT32_C(   551557753),  INT32_C(   271406663),  INT32_C(   827730343),
         INT32_C(  1877141046), -INT32_C(   933883159), -INT32_C(    68393394), -INT32_C(   593960736),  INT32_C(  1209237185), -INT32_C(  1440271773),  INT32_C(   800740488), -INT32_C(  1268707202) },
      { -INT32_C(  1656806467),  INT32_C(           0),  INT32_C(           0), -INT32_C(   701479952),  INT32_C(   961715231),  INT32_C(           0),  INT32_C(  1211640174),  INT32_C(           0),
         INT32_C(           0),  INT32_C(           0),  INT32_C(   849875900), -INT32_C(   774936669),  INT32_C(           0), -INT32_C(  1834124230), -INT32_C(   606595494), -INT32_C(  1135531766) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi32(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r = easysimd_mm512_maskz_dpwssd_epi32(test_vec[i].k, src, a, b);
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask16 k = easysimd_test_x86_random_mmask16();
    easysimd__m512i src = easysimd_test_x86_random_i32x16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    easysimd__m512i r = easysimd_mm512_maskz_dpwssd_epi32(k, src, a, b);

    easysimd_test_x86_write_mmask16(2, k, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, src, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

EASYSIMD_TEST_FUNC_LIST_BEGIN
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_dpwssd_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_dpwssd_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_dpwssd_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_dpwssd_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_dpwssd_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_dpwssd_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_dpwssd_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_dpwssd_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_dpwssd_epi32)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
