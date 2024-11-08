#define EASYSIMD_TEST_X86_AVX512_INSN dpbusd

#include <test/x86/avx512/test-avx512.h>
#include <easysimd/x86/avx512/dpbusd.h>

static int
test_easysimd_mm_dpbusd_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[4];
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { {  INT32_C(   323980410),  INT32_C(  1251535663), -INT32_C(    30467478),  INT32_C(   741325049) },
      { -INT32_C(  1157873637), -INT32_C(   442077135), -INT32_C(  1344200384),  INT32_C(   775580596) },
      {  INT32_C(   641829623), -INT32_C(   529409675), -INT32_C(   304176909),  INT32_C(  1947798872) },
      {  INT32_C(   323996063),  INT32_C(  1251548458), -INT32_C(    30482453),  INT32_C(   741349066) } },
    { {  INT32_C(  2133726542), -INT32_C(   966470266), -INT32_C(  1435154698), -INT32_C(  1445417039) },
      { -INT32_C(  1345381831), -INT32_C(   410042125),  INT32_C(   970223072), -INT32_C(   928125574) },
      { -INT32_C(  1991714045), -INT32_C(  1504727888), -INT32_C(  1538210574), -INT32_C(  1387452045) },
      {  INT32_C(  2133719830), -INT32_C(   966504575), -INT32_C(  1435152658), -INT32_C(  1445396571) } },
    { {  INT32_C(   912006211),  INT32_C(  1025370973), -INT32_C(   763956904), -INT32_C(   493149217) },
      { -INT32_C(  1351883777), -INT32_C(  2108245361), -INT32_C(   249125250),  INT32_C(   312374223) },
      { -INT32_C(   330696049),  INT32_C(  1026189029),  INT32_C(   923836504), -INT32_C(  1021727804) },
      {  INT32_C(   911980345),  INT32_C(  1025397626), -INT32_C(   763947889), -INT32_C(   493168560) } },
    { {  INT32_C(   494044302), -INT32_C(  1130379202),  INT32_C(  1051575663), -INT32_C(   934196168) },
      {  INT32_C(   750033478),  INT32_C(  1483333120),  INT32_C(  1133476223), -INT32_C(  1308186588) },
      {  INT32_C(  1792047148), -INT32_C(  1339658431),  INT32_C(  1844368437),  INT32_C(  1714765600) },
      {  INT32_C(   494061886), -INT32_C(  1130357610),  INT32_C(  1051581799), -INT32_C(   934165958) } },
    { { -INT32_C(   627905831),  INT32_C(  1194523848), -INT32_C(  1702182283), -INT32_C(  1756589974) },
      {  INT32_C(  1241586697), -INT32_C(  1040570228),  INT32_C(   472836348),  INT32_C(    25322536) },
      {  INT32_C(   383456590), -INT32_C(  2040656367),  INT32_C(   975235280), -INT32_C(  2100204167) },
      { -INT32_C(   627902950),  INT32_C(  1194526742), -INT32_C(  1702196851), -INT32_C(  1756580470) } },
    { {  INT32_C(   365745033), -INT32_C(   136919301), -INT32_C(   703396434), -INT32_C(  1210542743) },
      { -INT32_C(  1144147030), -INT32_C(  1857934399), -INT32_C(  1915985388),  INT32_C(  1494195663) },
      {  INT32_C(  1802427248),  INT32_C(  1331840417), -INT32_C(  1289325238), -INT32_C(  1251279349) },
      {  INT32_C(   365800009), -INT32_C(   136916746), -INT32_C(   703386593), -INT32_C(  1210545865) } },
    { {  INT32_C(  1936799665),  INT32_C(  1996796771), -INT32_C(   452669419), -INT32_C(   566357138) },
      { -INT32_C(  1890931474),  INT32_C(  1004449009),  INT32_C(   770573346), -INT32_C(  1260234750) },
      { -INT32_C(   215526512),  INT32_C(   476719878),  INT32_C(  1778478844),  INT32_C(  1867005825) },
      {  INT32_C(  1936788312),  INT32_C(  1996830797), -INT32_C(   452664107), -INT32_C(   566315596) } },
    { { -INT32_C(   570518805),  INT32_C(  1629019199), -INT32_C(   477231135), -INT32_C(   275287969) },
      { -INT32_C(   891109692), -INT32_C(   421114646),  INT32_C(   995157946),  INT32_C(   313169958) },
      {  INT32_C(  1794091051),  INT32_C(  1724581765),  INT32_C(  1850300686), -INT32_C(  1923227191) },
      { -INT32_C(   570509527),  INT32_C(  1629002219), -INT32_C(   477195642), -INT32_C(   275281031) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_x_mm_loadu_epi32(test_vec[i].src);
    easysimd__m128i a = easysimd_x_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r = easysimd_mm_dpbusd_epi32(src, a, b);
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_x_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m128i src = easysimd_test_x86_random_i32x4();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_dpbusd_epi32(src, a, b);

    easysimd_test_x86_write_i32x4(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x4(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x4(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm_mask_dpbusd_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[4];
    const easysimd__mmask8 k;
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { { -INT32_C(   679082603), -INT32_C(   185380171),  INT32_C(  1688544122), -INT32_C(  1638834310) },
      UINT8_C(153),
      {  INT32_C(   161296418), -INT32_C(  1713785207), -INT32_C(  1610920093),  INT32_C(     3413243) },
      { -INT32_C(   323613542),  INT32_C(   577153791), -INT32_C(  1348679088),  INT32_C(  1044921116) },
      { -INT32_C(   679097293), -INT32_C(   185380171),  INT32_C(  1688544122), -INT32_C(  1638822358) } },
    { { -INT32_C(   196614549),  INT32_C(     9314460),  INT32_C(  1805617520),  INT32_C(   929879197) },
      UINT8_C(224),
      { -INT32_C(   857791710),  INT32_C(  1394344329), -INT32_C(   646984802), -INT32_C(    96162284) },
      {  INT32_C(   378943733),  INT32_C(  1334220486),  INT32_C(   183300406),  INT32_C(  2146050909) },
      { -INT32_C(   196614549),  INT32_C(     9314460),  INT32_C(  1805617520),  INT32_C(   929879197) } },
    { { -INT32_C(   817051322),  INT32_C(  1747151050),  INT32_C(  1212256820),  INT32_C(   910329152) },
      UINT8_C(190),
      {  INT32_C(  1870941400), -INT32_C(  1012542254), -INT32_C(   467554368),  INT32_C(  1646960793) },
      {  INT32_C(  1429011180), -INT32_C(  1349937891),  INT32_C(  1559286230), -INT32_C(   350607853) },
      { -INT32_C(   817051322),  INT32_C(  1747099026),  INT32_C(  1212260979),  INT32_C(   910337173) } },
    { {  INT32_C(  1146789490),  INT32_C(   856162162),  INT32_C(  1192700078), -INT32_C(  1247198775) },
      UINT8_C( 59),
      {  INT32_C(  1800932054),  INT32_C(  1681983635),  INT32_C(   511155704),  INT32_C(  1452303287) },
      { -INT32_C(  1110911811),  INT32_C(    74185691),  INT32_C(  1422766866),  INT32_C(   848265820) },
      {  INT32_C(  1146762615),  INT32_C(   856164038),  INT32_C(  1192700078), -INT32_C(  1247206385) } },
    { {  INT32_C(   547219597), -INT32_C(   393879568),  INT32_C(   872873084),  INT32_C(   495621727) },
      UINT8_C(106),
      {  INT32_C(  1313200722), -INT32_C(   127907515), -INT32_C(  1722436586),  INT32_C(   740722500) },
      {  INT32_C(    68961829), -INT32_C(   931134261), -INT32_C(  1608010742), -INT32_C(  1861598146) },
      {  INT32_C(   547219597), -INT32_C(   393909109),  INT32_C(   872873084),  INT32_C(   495630619) } },
    { {  INT32_C(  1692356382), -INT32_C(  1352908903),  INT32_C(   944288244),  INT32_C(  1583640121) },
      UINT8_C(181),
      { -INT32_C(  2071960960), -INT32_C(  1752282910), -INT32_C(  1261097360), -INT32_C(  2016188872) },
      { -INT32_C(  2061486267),  INT32_C(  1165611155), -INT32_C(  2021740264), -INT32_C(  1791173611) },
      {  INT32_C(  1692358464), -INT32_C(  1352908903),  INT32_C(   944292356),  INT32_C(  1583640121) } },
    { {  INT32_C(   538557502),  INT32_C(  1974970117),  INT32_C(   220827093),  INT32_C(   966065395) },
      UINT8_C( 51),
      { -INT32_C(  2050507083), -INT32_C(   392360905),  INT32_C(  1727865994),  INT32_C(   497324640) },
      {  INT32_C(  1378010283),  INT32_C(   153589628), -INT32_C(  1107544896),  INT32_C(  2112894408) },
      {  INT32_C(   538548389),  INT32_C(  1974983888),  INT32_C(   220827093),  INT32_C(   966065395) } },
    { {  INT32_C(   704821235),  INT32_C(  1293066435),  INT32_C(   632491972),  INT32_C(  1279416225) },
      UINT8_C( 28),
      { -INT32_C(    73884060), -INT32_C(   105143867), -INT32_C(   742229859),  INT32_C(   549863273) },
      { -INT32_C(   488312510),  INT32_C(   329658627),  INT32_C(  1018481637),  INT32_C(  1901592845) },
      {  INT32_C(   704821235),  INT32_C(  1293066435),  INT32_C(   632479289),  INT32_C(  1279438693) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_x_mm_loadu_epi32(test_vec[i].src);
    easysimd__m128i a = easysimd_x_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r = easysimd_mm_mask_dpbusd_epi32(src, test_vec[i].k, a, b);
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
    easysimd__m128i r = easysimd_mm_mask_dpbusd_epi32(src, k, a, b);

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
test_easysimd_mm_maskz_dpbusd_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int32_t src[4];
    const int32_t a[4];
    const int32_t b[4];
    const int32_t r[4];
  } test_vec[] = {
    { UINT8_C(159),
      { -INT32_C(  1855623952), -INT32_C(  1607508440),  INT32_C(  1611203104), -INT32_C(  1180554552) },
      {  INT32_C(  1069384718), -INT32_C(   165359574), -INT32_C(  2063375996), -INT32_C(  1440385607) },
      {  INT32_C(   154897121),  INT32_C(   162163432),  INT32_C(   896119660), -INT32_C(   336720931) },
      { -INT32_C(  1855628244), -INT32_C(  1607488282),  INT32_C(  1611217255), -INT32_C(  1180563976) } },
    { UINT8_C(143),
      {  INT32_C(  2075732907), -INT32_C(  1342132401),  INT32_C(   678069683),  INT32_C(   873010346) },
      { -INT32_C(  1172499633),  INT32_C(  1881548477), -INT32_C(  1706140785), -INT32_C(   181847734) },
      { -INT32_C(  1267604892),  INT32_C(  1164210578), -INT32_C(  1586639114),  INT32_C(   852850402) },
      {  INT32_C(  2075729378), -INT32_C(  1342137369),  INT32_C(   678057575),  INT32_C(   873025339) } },
    { UINT8_C(137),
      {  INT32_C(   407301362),  INT32_C(  1856485138),  INT32_C(  1052262661),  INT32_C(  1302572394) },
      { -INT32_C(  1881188578), -INT32_C(  1970920261),  INT32_C(   124528529), -INT32_C(   275669252) },
      { -INT32_C(  1660430454), -INT32_C(  1827950706),  INT32_C(  1540539376), -INT32_C(  1884785296) },
      {  INT32_C(   407281659),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1302579337) } },
    { UINT8_C(204),
      { -INT32_C(  1417208185), -INT32_C(   885255772),  INT32_C(   482886526), -INT32_C(  1398294572) },
      {  INT32_C(  1832535230),  INT32_C(   308203087),  INT32_C(   360888736), -INT32_C(   387903135) },
      { -INT32_C(   728537040), -INT32_C(   123678854), -INT32_C(   418093038), -INT32_C(   577520865) },
      {  INT32_C(           0),  INT32_C(           0),  INT32_C(   482910721), -INT32_C(  1398325383) } },
    {    UINT8_MAX,
      { -INT32_C(  1689367603),  INT32_C(  1648058537), -INT32_C(   188526365), -INT32_C(  1708872911) },
      {  INT32_C(   219478334),  INT32_C(    18812057), -INT32_C(   601881056),  INT32_C(  1742470553) },
      { -INT32_C(   251516344),  INT32_C(  1834172042), -INT32_C(  1067313522), -INT32_C(    10844479) },
      { -INT32_C(  1689352836),  INT32_C(  1648043909), -INT32_C(   188540825), -INT32_C(  1708893809) } },
    { UINT8_C(127),
      {  INT32_C(  2048396398),  INT32_C(   848959788),  INT32_C(   936146489),  INT32_C(  2088710994) },
      {  INT32_C(  1929802037), -INT32_C(   620596028),  INT32_C(  1553777366), -INT32_C(  1965253604) },
      { -INT32_C(   737872728),  INT32_C(  1191681550),  INT32_C(  1736364821),  INT32_C(  1004797446) },
      {  INT32_C(  2048385342),  INT32_C(   848966727),  INT32_C(   936171385),  INT32_C(  2088712612) } },
    { UINT8_C(111),
      {  INT32_C(  1563668457),  INT32_C(  1932725937),  INT32_C(  1200591019),  INT32_C(  1626282348) },
      { -INT32_C(  1150368739), -INT32_C(  1630488885),  INT32_C(   849622836),  INT32_C(    77717274) },
      {  INT32_C(  1063376270), -INT32_C(  1900898845), -INT32_C(  1848295131),  INT32_C(  2012333402) },
      {  INT32_C(  1563679174),  INT32_C(  1932666340),  INT32_C(  1200583916),  INT32_C(  1626269592) } },
    { UINT8_C(137),
      {  INT32_C(   374616928),  INT32_C(   994767363),  INT32_C(  1968536982), -INT32_C(   217818850) },
      {  INT32_C(  1356284859),  INT32_C(   930440694),  INT32_C(     9569851),  INT32_C(  1468598775) },
      {  INT32_C(  1080941884),  INT32_C(  1702607055),  INT32_C(  1406849077), -INT32_C(   448274902) },
      {  INT32_C(   374654358),  INT32_C(           0),  INT32_C(           0), -INT32_C(   217801404) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m128i src = easysimd_x_mm_loadu_epi32(test_vec[i].src);
    easysimd__m128i a = easysimd_x_mm_loadu_epi32(test_vec[i].a);
    easysimd__m128i b = easysimd_x_mm_loadu_epi32(test_vec[i].b);
    easysimd__m128i r = easysimd_mm_maskz_dpbusd_epi32(test_vec[i].k, src, a, b);
    easysimd_test_x86_assert_equal_i32x4(r, easysimd_x_mm_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__mmask8 k = easysimd_test_x86_random_mmask8();
    easysimd__m128i src = easysimd_test_x86_random_i32x4();
    easysimd__m128i a = easysimd_test_x86_random_i32x4();
    easysimd__m128i b = easysimd_test_x86_random_i32x4();
    easysimd__m128i r = easysimd_mm_maskz_dpbusd_epi32(k, src, a, b);

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
test_easysimd_mm256_dpbusd_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[8];
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { {  INT32_C(   450888026),  INT32_C(   151496528),  INT32_C(   741466425),  INT32_C(    43369007), -INT32_C(   360831191), -INT32_C(   163978821),  INT32_C(  1354910642),  INT32_C(   329181624) },
      {  INT32_C(   992837355),  INT32_C(  1598305318),  INT32_C(  1166832917),  INT32_C(  1632051511),  INT32_C(    55297352),  INT32_C(  1492812966), -INT32_C(  2002142000), -INT32_C(  1868871771) },
      { -INT32_C(   338966075),  INT32_C(   323686397), -INT32_C(  1135028348),  INT32_C(  1075683320),  INT32_C(   189032549), -INT32_C(  1117503763), -INT32_C(  1622864390),  INT32_C(   422633812) },
      {  INT32_C(   450863607),  INT32_C(   151504099),  INT32_C(   741466652),  INT32_C(    43373633), -INT32_C(   360798298), -INT32_C(   163954687),  INT32_C(  1354910307),  INT32_C(   329204323) } },
    { { -INT32_C(  1492780118), -INT32_C(  1883615221),  INT32_C(   525013543),  INT32_C(   392128690), -INT32_C(  1105026095), -INT32_C(   612661535), -INT32_C(   411320173),  INT32_C(  1258400673) },
      { -INT32_C(  1309473114),  INT32_C(  2101456214),  INT32_C(  1922862272), -INT32_C(   964035595), -INT32_C(  2138788961), -INT32_C(   983761102),  INT32_C(  1638782911),  INT32_C(   682405506) },
      {  INT32_C(   182099892),  INT32_C(   226958157), -INT32_C(  1669389401), -INT32_C(  1117648866), -INT32_C(   448862541), -INT32_C(  1515480347), -INT32_C(   217687951), -INT32_C(  1172590074) },
      { -INT32_C(  1492800780), -INT32_C(  1883610168),  INT32_C(   525009767),  INT32_C(   392138208), -INT32_C(  1105038056), -INT32_C(   612714642), -INT32_C(   411279893),  INT32_C(  1258389725) } },
    { { -INT32_C(  1614482094), -INT32_C(  1196668144), -INT32_C(  1940640914), -INT32_C(   431311053), -INT32_C(  2083747683), -INT32_C(  1826064606), -INT32_C(   712626481),  INT32_C(   848273888) },
      { -INT32_C(  1479453801),  INT32_C(   207584670), -INT32_C(   610749272),  INT32_C(   146989675), -INT32_C(  1937011094), -INT32_C(   736120059), -INT32_C(  1029069343), -INT32_C(   554420153) },
      {  INT32_C(   696632971), -INT32_C(   332012221),  INT32_C(    80203161),  INT32_C(   437029296),  INT32_C(   480745495),  INT32_C(   770754123), -INT32_C(  1276143252),  INT32_C(  1553064913) },
      { -INT32_C(  1614523435), -INT32_C(  1196656138), -INT32_C(  1940675186), -INT32_C(   431343971), -INT32_C(  2083768464), -INT32_C(  1826065569), -INT32_C(   712636987),  INT32_C(   848262267) } },
    { { -INT32_C(   310045015), -INT32_C(  1797670149),  INT32_C(   949526664),  INT32_C(  1095935274), -INT32_C(  2007041731),  INT32_C(   750079680), -INT32_C(  1176525592),  INT32_C(   823554184) },
      { -INT32_C(  2111923322), -INT32_C(   568920234), -INT32_C(  1038700648), -INT32_C(  1862047660),  INT32_C(   572088674), -INT32_C(  1739665488), -INT32_C(    78566029),  INT32_C(   590178205) },
      {  INT32_C(  1504004867), -INT32_C(   617104317), -INT32_C(  1080210069),  INT32_C(   391159989), -INT32_C(  1304860414), -INT32_C(  1404401864),  INT32_C(  1369938868),  INT32_C(   108385283) },
      { -INT32_C(   310024148), -INT32_C(  1797688187),  INT32_C(   949541615),  INT32_C(  1095922565), -INT32_C(  2007032577),  INT32_C(   750057493), -INT32_C(  1176525755),  INT32_C(   823555598) } },
    { {  INT32_C(  1667176992),  INT32_C(  1094620886), -INT32_C(  1744774173),  INT32_C(  2108706939), -INT32_C(   231740998),  INT32_C(   631142769),  INT32_C(   410469909),  INT32_C(   975105050) },
      { -INT32_C(   576881402), -INT32_C(   165749997),  INT32_C(   848175030),  INT32_C(   699350639), -INT32_C(  1726161112),  INT32_C(  1841216088),  INT32_C(   461714688),  INT32_C(   676700961) },
      {  INT32_C(   856027936), -INT32_C(  2060901426), -INT32_C(  1296582590),  INT32_C(   517695222), -INT32_C(  1632110779), -INT32_C(  1307871566), -INT32_C(   842165845),  INT32_C(  1442128692) },
      {  INT32_C(  1667187615),  INT32_C(  1094598573), -INT32_C(  1744778659),  INT32_C(  2108706908), -INT32_C(   231757283),  INT32_C(   631151441),  INT32_C(   410455866),  INT32_C(   975114936) } },
    { { -INT32_C(   460785130),  INT32_C(  1617539613),  INT32_C(  1611800682), -INT32_C(   864031353), -INT32_C(  1754646811),  INT32_C(  1481209516),  INT32_C(   992286471),  INT32_C(  1351621178) },
      {  INT32_C(   825497876),  INT32_C(   898735819),  INT32_C(  1167434686),  INT32_C(  1980896401), -INT32_C(   150111157), -INT32_C(   112240910), -INT32_C(  1489669011), -INT32_C(  1560820338) },
      { -INT32_C(  1445712674), -INT32_C(  1998690870), -INT32_C(  1697745911), -INT32_C(   753868664),  INT32_C(  1338711388), -INT32_C(   515368333),  INT32_C(   495484303),  INT32_C(   566263875) },
      { -INT32_C(   460791261),  INT32_C(  1617533319),  INT32_C(  1611806812), -INT32_C(   864054415), -INT32_C(  1754617491),  INT32_C(  1481237551),  INT32_C(   992287137),  INT32_C(  1351595014) } },
    { {  INT32_C(  1993053356),  INT32_C(    50244089), -INT32_C(  1499673571),  INT32_C(   158969004),  INT32_C(  1029194953), -INT32_C(   299982753),  INT32_C(  1628153374), -INT32_C(   763180250) },
      {  INT32_C(  1497910623),  INT32_C(   341526519), -INT32_C(  1061488621),  INT32_C(  1858679972), -INT32_C(   676650632), -INT32_C(   540685887), -INT32_C(  1774136977), -INT32_C(    60243044) },
      {  INT32_C(   123056400),  INT32_C(   186429688),  INT32_C(  1305204392), -INT32_C(  2084858870),  INT32_C(  2002413237), -INT32_C(  1638522833), -INT32_C(  1925933071),  INT32_C(  1787403610) },
      {  INT32_C(  1993055536),  INT32_C(    50239201), -INT32_C(  1499680733),  INT32_C(   158937409),  INT32_C(  1029230294), -INT32_C(   299972162),  INT32_C(  1628115842), -INT32_C(   763171179) } },
    { {  INT32_C(  1181933134),  INT32_C(   928091791),  INT32_C(  1870929252),  INT32_C(  1743929265), -INT32_C(   723628891),  INT32_C(  1567831148),  INT32_C(   636200907), -INT32_C(  1836026812) },
      { -INT32_C(   489160109), -INT32_C(   199677296), -INT32_C(   127689145), -INT32_C(  2090904099),  INT32_C(   240598434),  INT32_C(  1030539890), -INT32_C(  1235003534),  INT32_C(   508163019) },
      { -INT32_C(  2063588875), -INT32_C(  1837557173), -INT32_C(  1785995848), -INT32_C(   736564686), -INT32_C(  1696370648), -INT32_C(  1395175622),  INT32_C(  1902328486),  INT32_C(   579906605) },
      {  INT32_C(  1181904489),  INT32_C(   928079868),  INT32_C(  1870880491),  INT32_C(  1743934961), -INT32_C(   723619530),  INT32_C(  1567844166),  INT32_C(   636226060), -INT32_C(  1836045245) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_x_mm256_loadu_epi32(test_vec[i].src);
    easysimd__m256i a = easysimd_x_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_x_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r = easysimd_mm256_dpbusd_epi32(src, a, b);
    easysimd_test_x86_assert_equal_i32x8(r, easysimd_x_mm256_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m256i src = easysimd_test_x86_random_i32x8();
    easysimd__m256i a = easysimd_test_x86_random_i32x8();
    easysimd__m256i b = easysimd_test_x86_random_i32x8();
    easysimd__m256i r = easysimd_mm256_dpbusd_epi32(src, a, b);

    easysimd_test_x86_write_i32x8(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x8(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x8(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm256_mask_dpbusd_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[8];
    const easysimd__mmask8 k;
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { { -INT32_C(  1271476468), -INT32_C(   844146845), -INT32_C(  2084586903), -INT32_C(   190964784),  INT32_C(  1371235618), -INT32_C(  1050571527),  INT32_C(   555037273), -INT32_C(    18113039) },
      UINT8_C(108),
      {  INT32_C(  2026877473),  INT32_C(   467770465),  INT32_C(  2011915356),  INT32_C(  1754914563),  INT32_C(   660728474),  INT32_C(  2122326860), -INT32_C(   714104520), -INT32_C(  1388155508) },
      { -INT32_C(  2145054433),  INT32_C(   177931950),  INT32_C(  1853982315),  INT32_C(    14031717),  INT32_C(  1361524741), -INT32_C(  1815107749), -INT32_C(   714522808), -INT32_C(   863851604) },
      { -INT32_C(  1271476468), -INT32_C(   844146845), -INT32_C(  2084606014), -INT32_C(   190964886),  INT32_C(  1371235618), -INT32_C(  1050587732),  INT32_C(   555053944), -INT32_C(    18113039) } },
    { {  INT32_C(  1783408828),  INT32_C(   427092142), -INT32_C(   729287058),  INT32_C(   383016465), -INT32_C(   244778090), -INT32_C(   360433758),  INT32_C(   599780726),  INT32_C(  1441743512) },
      UINT8_C(234),
      {  INT32_C(   597212987),  INT32_C(   714256948), -INT32_C(  1757714887),  INT32_C(   892162362), -INT32_C(   237560135), -INT32_C(  1872248413), -INT32_C(  1020687743), -INT32_C(  1246921095) },
      {  INT32_C(  1909998909),  INT32_C(   832269047),  INT32_C(   180934352), -INT32_C(   515901912), -INT32_C(  1210968556),  INT32_C(  1497905880),  INT32_C(  1042051524),  INT32_C(   754174447) },
      {  INT32_C(  1783408828),  INT32_C(   427097854), -INT32_C(   729287058),  INT32_C(   383019167), -INT32_C(   244778090), -INT32_C(   360408852),  INT32_C(   599801818),  INT32_C(  1441740240) } },
    { {  INT32_C(   111069966),  INT32_C(   104282422),  INT32_C(   940703504),  INT32_C(   169431285),  INT32_C(  1069673575), -INT32_C(   359134938),  INT32_C(  1764275322), -INT32_C(  1953096835) },
      UINT8_C(231),
      {  INT32_C(  1830654260), -INT32_C(   931322936), -INT32_C(  2051164876), -INT32_C(  1142110257), -INT32_C(  1830737015),  INT32_C(  1997392835),  INT32_C(   267679476),  INT32_C(  1089896204) },
      { -INT32_C(   642903279),  INT32_C(  1805724471), -INT32_C(  1326424095), -INT32_C(  1351885786), -INT32_C(   901624825),  INT32_C(   188829463), -INT32_C(   769968699), -INT32_C(   971894347) },
      {  INT32_C(   111066976),  INT32_C(   104304452),  INT32_C(   940705423),  INT32_C(   169431285),  INT32_C(  1069673575), -INT32_C(   359112262),  INT32_C(  1764273078), -INT32_C(  1953094860) } },
    { {  INT32_C(  1553973285), -INT32_C(   842579476), -INT32_C(   964839264), -INT32_C(  1669928812),  INT32_C(  1265023028), -INT32_C(   866670585), -INT32_C(  1835109667), -INT32_C(  1470582397) },
      UINT8_C(113),
      {  INT32_C(   962397432), -INT32_C(  2066142516), -INT32_C(  1894211673), -INT32_C(   842812395), -INT32_C(  1009512677), -INT32_C(   660561562), -INT32_C(   262458561), -INT32_C(  2090728309) },
      { -INT32_C(   725827832), -INT32_C(  1856465430),  INT32_C(  1243705653),  INT32_C(  1075373093),  INT32_C(  1493429491), -INT32_C(   852384627),  INT32_C(  1656589783), -INT32_C(  1712971887) },
      {  INT32_C(  1553966177), -INT32_C(   842579476), -INT32_C(   964839264), -INT32_C(  1669928812),  INT32_C(  1265040368), -INT32_C(   866700303), -INT32_C(  1835100692), -INT32_C(  1470582397) } },
    { { -INT32_C(   915496225),  INT32_C(  1834665528),  INT32_C(  1572305719),  INT32_C(  1402851168),  INT32_C(  1236115900),  INT32_C(   471260741), -INT32_C(    58796949), -INT32_C(   761895693) },
      UINT8_C(  7),
      { -INT32_C(   901801212),  INT32_C(  1896000758),  INT32_C(   869359459), -INT32_C(  1645271556), -INT32_C(  1327351598),  INT32_C(   589102671), -INT32_C(   518645635), -INT32_C(  1293358674) },
      {  INT32_C(  2055022468),  INT32_C(   938246099), -INT32_C(   630538786), -INT32_C(  1267181086), -INT32_C(   530228591), -INT32_C(   704413351),  INT32_C(  1219959449), -INT32_C(  2013618173) },
      { -INT32_C(   915458157),  INT32_C(  1834682477),  INT32_C(  1572316204),  INT32_C(  1402851168),  INT32_C(  1236115900),  INT32_C(   471260741), -INT32_C(    58796949), -INT32_C(   761895693) } },
    { { -INT32_C(  1694337081), -INT32_C(   724373770), -INT32_C(  1901118293),  INT32_C(   641869717), -INT32_C(   637032575), -INT32_C(  1028650456),  INT32_C(   654993444), -INT32_C(   810613752) },
      UINT8_C(124),
      { -INT32_C(  1619891535),  INT32_C(  2018133820),  INT32_C(   487446774), -INT32_C(  1029819365),  INT32_C(  1173059899), -INT32_C(  1855279831), -INT32_C(  1147563593), -INT32_C(   248026816) },
      {  INT32_C(   277916371), -INT32_C(   410461199), -INT32_C(   838560077),  INT32_C(    93430474),  INT32_C(  1162509339), -INT32_C(   522800087), -INT32_C(  2036633530), -INT32_C(  1401367591) },
      { -INT32_C(  1694337081), -INT32_C(   724373770), -INT32_C(  1901161529),  INT32_C(   641846803), -INT32_C(   636993827), -INT32_C(  1028671015),  INT32_C(   654984227), -INT32_C(   810613752) } },
    { {  INT32_C(  1857816701), -INT32_C(  1756019229), -INT32_C(  1520084517),  INT32_C(   397080315), -INT32_C(  1688406926), -INT32_C(   277138775),  INT32_C(  2071271330),  INT32_C(  1730735594) },
      UINT8_C(246),
      {  INT32_C(   702141924), -INT32_C(  2096861142), -INT32_C(   864114218),  INT32_C(  1212126547),  INT32_C(   619830001),  INT32_C(  1841750102),  INT32_C(  1129792085),  INT32_C(  1312407146) },
      { -INT32_C(  1116204397),  INT32_C(  1497398403),  INT32_C(  2032516646),  INT32_C(  1170302036), -INT32_C(  1788235201), -INT32_C(   402509678), -INT32_C(   601138830), -INT32_C(  1440062441) },
      {  INT32_C(  1857816701), -INT32_C(  1755998676), -INT32_C(  1520058193),  INT32_C(   397080315), -INT32_C(  1688387294), -INT32_C(   277139703),  INT32_C(  2071288223),  INT32_C(  1730752718) } },
    { { -INT32_C(   110648714),  INT32_C(  1163044639), -INT32_C(  1178699675),  INT32_C(   469729500), -INT32_C(   978294734),  INT32_C(   179155608),  INT32_C(   585553931), -INT32_C(  1295249092) },
      UINT8_C(179),
      { -INT32_C(   623727565),  INT32_C(  1983911934),  INT32_C(  1448344022),  INT32_C(  1636331256), -INT32_C(   772190945), -INT32_C(   757267206), -INT32_C(    82903062), -INT32_C(     5258804) },
      {  INT32_C(  1809482093),  INT32_C(  1877088921),  INT32_C(   214250771), -INT32_C(  1033024093), -INT32_C(  1785436517),  INT32_C(  1432908138),  INT32_C(  1011906160), -INT32_C(  1539506377) },
      { -INT32_C(   110649653),  INT32_C(  1163030253), -INT32_C(  1178699675),  INT32_C(   469729500), -INT32_C(   978339266),  INT32_C(   179223394),  INT32_C(   585553931), -INT32_C(  1295251025) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_x_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_x_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_x_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r = easysimd_mm256_mask_dpbusd_epi32(src, k, a, b);
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
    easysimd__m256i r = easysimd_mm256_mask_dpbusd_epi32(src, k, a, b);

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
test_easysimd_mm256_maskz_dpbusd_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask8 k;
    const int32_t src[8];
    const int32_t a[8];
    const int32_t b[8];
    const int32_t r[8];
  } test_vec[] = {
    { UINT8_C(129),
      {  INT32_C(   807014422),  INT32_C(   658737650), -INT32_C(  1681240242),  INT32_C(   590777788), -INT32_C(  1836200927), -INT32_C(  1442651596),  INT32_C(   853688115), -INT32_C(  1850505605) },
      { -INT32_C(  2000564842), -INT32_C(  1532033962),  INT32_C(   272661076),  INT32_C(   674461191),  INT32_C(  1991950658), -INT32_C(   702432093),  INT32_C(  1980302331),  INT32_C(   537377929) },
      { -INT32_C(   525809526),  INT32_C(   562387149), -INT32_C(   651049518),  INT32_C(  2114020667), -INT32_C(   906708186),  INT32_C(  1939871095), -INT32_C(  1561745128), -INT32_C(   272436891) },
      {  INT32_C(   806963906),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1850495566) } },
    { UINT8_C(185),
      { -INT32_C(  1031352214),  INT32_C(   445949781),  INT32_C(  1045786073),  INT32_C(   694473582), -INT32_C(   576639544),  INT32_C(  1995838669), -INT32_C(   287532803), -INT32_C(   995570854) },
      { -INT32_C(   259576165), -INT32_C(  1358292010), -INT32_C(   135438199), -INT32_C(    81702605),  INT32_C(  1289339518), -INT32_C(   742207530), -INT32_C(  1044210073),  INT32_C(    92629610) },
      {  INT32_C(  1861618840), -INT32_C(  1323499480), -INT32_C(  1800926624), -INT32_C(   628111013),  INT32_C(  1629907083), -INT32_C(  1640634313), -INT32_C(   245368953), -INT32_C(   101259935) },
      { -INT32_C(  1031342732),  INT32_C(           0),  INT32_C(           0),  INT32_C(   694460501), -INT32_C(   576618492),  INT32_C(  1995835075),  INT32_C(           0), -INT32_C(   995564799) } },
    { UINT8_C(241),
      { -INT32_C(   317101844), -INT32_C(  1873950075),  INT32_C(  1022091635), -INT32_C(   641153679), -INT32_C(   737072661),  INT32_C(  1432072030), -INT32_C(   189379569),  INT32_C(   803582018) },
      { -INT32_C(  1659044072),  INT32_C(  1009609161), -INT32_C(  1132914357), -INT32_C(   896187938), -INT32_C(   912349590),  INT32_C(  1696528726), -INT32_C(  2024155835), -INT32_C(  1649000827) },
      {  INT32_C(   121295421), -INT32_C(  2025625540),  INT32_C(  1598274689),  INT32_C(  1747572989), -INT32_C(   718157953),  INT32_C(    87707584), -INT32_C(  1450339548),  INT32_C(   256263121) },
      { -INT32_C(   317109387),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   737069562),  INT32_C(  1432088442), -INT32_C(   189422282),  INT32_C(   803595016) } },
    { UINT8_C( 22),
      { -INT32_C(   380496255),  INT32_C(   376101209), -INT32_C(   200029924), -INT32_C(  1166836749), -INT32_C(    59029332),  INT32_C(   354451586),  INT32_C(  1374145037), -INT32_C(  1855457776) },
      {  INT32_C(  1702541580), -INT32_C(  1367612270), -INT32_C(  1583181906), -INT32_C(  1218767350), -INT32_C(   508307874),  INT32_C(  1693897559), -INT32_C(  1363812963), -INT32_C(   549511981) },
      {  INT32_C(  1732623061),  INT32_C(  1276493982),  INT32_C(  1525528655),  INT32_C(   739330510),  INT32_C(  1980613663),  INT32_C(   903546007), -INT32_C(  1243377439), -INT32_C(  2104221011) },
      {  INT32_C(           0),  INT32_C(   376088116), -INT32_C(   200015062),  INT32_C(           0), -INT32_C(    59010381),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(220),
      { -INT32_C(  1703154983), -INT32_C(  1209415681), -INT32_C(    24820811),  INT32_C(   404599380), -INT32_C(  1011837761), -INT32_C(     5905041), -INT32_C(   357803320), -INT32_C(   926470162) },
      {  INT32_C(   409092633), -INT32_C(  1093645559), -INT32_C(   474196593), -INT32_C(   939730425), -INT32_C(   578114450),  INT32_C(  1507602321),  INT32_C(  2017692041), -INT32_C(   784332104) },
      {  INT32_C(  1458152012),  INT32_C(  2081733101),  INT32_C(   375443727),  INT32_C(   433937579), -INT32_C(  1711904760),  INT32_C(   569561751),  INT32_C(   328807771), -INT32_C(  1931159232) },
      {  INT32_C(           0),  INT32_C(           0), -INT32_C(    24799619),  INT32_C(   404614996), -INT32_C(  1011843136),  INT32_C(           0), -INT32_C(   357788213), -INT32_C(   926484812) } },
    { UINT8_C(123),
      { -INT32_C(  2023169330), -INT32_C(   929634825), -INT32_C(  1586254523), -INT32_C(   223769462),  INT32_C(  1418347138),  INT32_C(  1773185844),  INT32_C(   497664836),  INT32_C(  1989686952) },
      {  INT32_C(   268238872),  INT32_C(   718771429), -INT32_C(   875869631),  INT32_C(  1488876758), -INT32_C(   357742410),  INT32_C(   928275955), -INT32_C(   933954272),  INT32_C(  1279192115) },
      { -INT32_C(   765772564),  INT32_C(   301741008),  INT32_C(  1423755389), -INT32_C(   206792132), -INT32_C(   706913822), -INT32_C(   687066698),  INT32_C(  1654612015),  INT32_C(   967761484) },
      { -INT32_C(  2023147477), -INT32_C(   929638415),  INT32_C(           0), -INT32_C(   223785558),  INT32_C(  1418331969),  INT32_C(  1773171172),  INT32_C(   497702080),  INT32_C(           0) } },
    { UINT8_C( 26),
      {  INT32_C(  1038748426), -INT32_C(   809829625),  INT32_C(  1913392855),  INT32_C(   341114811),  INT32_C(   248195804), -INT32_C(  1757568458),  INT32_C(   535011137),  INT32_C(  1480137806) },
      {  INT32_C(   798303015), -INT32_C(   151105762),  INT32_C(   426248798), -INT32_C(   433210359),  INT32_C(   502593766), -INT32_C(   592170598),  INT32_C(   519804880), -INT32_C(   613009996) },
      {  INT32_C(  1997146968), -INT32_C(  1184036517),  INT32_C(   500356371),  INT32_C(  2013527953), -INT32_C(  1835665416), -INT32_C(   126990040), -INT32_C(  1810404640), -INT32_C(   143618657) },
      {  INT32_C(           0), -INT32_C(   809815964),  INT32_C(           0),  INT32_C(   341141359),  INT32_C(   248162434),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT8_C(153),
      { -INT32_C(  2081132934), -INT32_C(  1332236837),  INT32_C(  2118300799), -INT32_C(  1367950665), -INT32_C(  1730803377), -INT32_C(   512176521),  INT32_C(  1937772005), -INT32_C(   150177667) },
      { -INT32_C(  1032126234),  INT32_C(   745673389),  INT32_C(  2108339398), -INT32_C(  1121246866), -INT32_C(  1588264662), -INT32_C(  1249653041),  INT32_C(  1479017435),  INT32_C(  1649357947) },
      { -INT32_C(   517682636), -INT32_C(  1576167716), -INT32_C(  1189103797),  INT32_C(    41372376),  INT32_C(   463785035),  INT32_C(  1976575898), -INT32_C(  1496451030),  INT32_C(  1611144492) },
      { -INT32_C(  2081122560),  INT32_C(           0),  INT32_C(           0), -INT32_C(  1367947128), -INT32_C(  1730803752),  INT32_C(           0),  INT32_C(           0), -INT32_C(   150160707) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m256i src = easysimd_x_mm256_loadu_epi32(test_vec[i].src);
    easysimd__mmask8 k = test_vec[i].k;
    easysimd__m256i a = easysimd_x_mm256_loadu_epi32(test_vec[i].a);
    easysimd__m256i b = easysimd_x_mm256_loadu_epi32(test_vec[i].b);
    easysimd__m256i r = easysimd_mm256_maskz_dpbusd_epi32(k, src, a, b);
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
    easysimd__m256i r = easysimd_mm256_maskz_dpbusd_epi32(k, src, a, b);

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
test_easysimd_mm512_dpbusd_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[16];
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { { -INT32_C(  1288507764), -INT32_C(  1804139285),  INT32_C(   198043533), -INT32_C(  1347328338), -INT32_C(  1619901069),  INT32_C(   898564358), -INT32_C(   195693852), -INT32_C(   898142146),
         INT32_C(  1249815134),  INT32_C(  1004467630), -INT32_C(  1975079716), -INT32_C(  2059798510),  INT32_C(  1143319613), -INT32_C(  1854294867), -INT32_C(   460992602), -INT32_C(  1632633537) },
      {  INT32_C(  1441279399), -INT32_C(    24066526), -INT32_C(  2054563982),  INT32_C(   202031823),  INT32_C(   475017070), -INT32_C(  1968321820), -INT32_C(   663800935), -INT32_C(   680124880),
         INT32_C(  1831624267), -INT32_C(  1754547163),  INT32_C(  1646064787),  INT32_C(   628041655),  INT32_C(   977387350),  INT32_C(   583397257),  INT32_C(  1392194594), -INT32_C(  1658228398) },
      { -INT32_C(   183872048), -INT32_C(  1517521647),  INT32_C(   554150250),  INT32_C(   642152144), -INT32_C(  1084127178), -INT32_C(  1713232265), -INT32_C(  1393828518),  INT32_C(   508105806),
         INT32_C(  2048086889),  INT32_C(   840933576),  INT32_C(   424879945), -INT32_C(   750806371), -INT32_C(  1718378462),  INT32_C(   540177862), -INT32_C(  1597235886), -INT32_C(  1698753231) },
      { -INT32_C(  1288510570), -INT32_C(  1804155359),  INT32_C(   198042260), -INT32_C(  1347314226), -INT32_C(  1619894829),  INT32_C(   898579762), -INT32_C(   195702342), -INT32_C(   898122738),
         INT32_C(  1249844945),  INT32_C(  1004458377), -INT32_C(  1975054695), -INT32_C(  2059815277),  INT32_C(  1143291142), -INT32_C(  1854263912), -INT32_C(   461009230), -INT32_C(  1632645825) } },
    { {  INT32_C(   823448168), -INT32_C(  1151126414), -INT32_C(   120277157),  INT32_C(  1942754385),  INT32_C(  2064408500),  INT32_C(   647708372), -INT32_C(  1933154213), -INT32_C(   433683075),
        -INT32_C(   904447400), -INT32_C(   913933714), -INT32_C(  2101192143),  INT32_C(   603295342), -INT32_C(  1046609427), -INT32_C(  1696122561),  INT32_C(   522628513), -INT32_C(  1962587085) },
      { -INT32_C(   178971513), -INT32_C(   926950506), -INT32_C(  1538621130), -INT32_C(    54050801), -INT32_C(  2118294207),  INT32_C(  1075553439), -INT32_C(  2074131887),  INT32_C(   353330318),
         INT32_C(   386622848),  INT32_C(  1994377792),  INT32_C(  1511729483), -INT32_C(  1420369303), -INT32_C(   416541880),  INT32_C(   136791991),  INT32_C(   411928457),  INT32_C(  1814928619) },
      {  INT32_C(  1115895809),  INT32_C(  1303929346), -INT32_C(   173550709), -INT32_C(    23003722), -INT32_C(   941241328), -INT32_C(  1647309805),  INT32_C(  2142592403), -INT32_C(    68427015),
         INT32_C(   490565147),  INT32_C(  1533801936),  INT32_C(  2135954121),  INT32_C(   545124368), -INT32_C(   790076740),  INT32_C(    40745070),  INT32_C(   243343893),  INT32_C(   520711172) },
      {  INT32_C(   823455416), -INT32_C(  1151103004), -INT32_C(   120297670),  INT32_C(  1942733475),  INT32_C(  2064391832),  INT32_C(   647705729), -INT32_C(  1933147265), -INT32_C(   433687489),
        -INT32_C(   904431496), -INT32_C(   913904409), -INT32_C(  2101181940),  INT32_C(   603309628), -INT32_C(  1046624605), -INT32_C(  1696103276),  INT32_C(   522618409), -INT32_C(  1962565544) } },
    { { -INT32_C(  1438824742),  INT32_C(    84256828), -INT32_C(   897296710),  INT32_C(    65732934), -INT32_C(   774646941), -INT32_C(  1596768117),  INT32_C(  1722700898), -INT32_C(  1702446912),
         INT32_C(   977585150),  INT32_C(   624904811), -INT32_C(   420428896),  INT32_C(   669637572),  INT32_C(   972668078),  INT32_C(  1591332092), -INT32_C(   507148511),  INT32_C(  1048333119) },
      {  INT32_C(  2037956622), -INT32_C(  1432438774),  INT32_C(  1083281019),  INT32_C(   392657513),  INT32_C(   860905527),  INT32_C(  1318201645), -INT32_C(   265332815), -INT32_C(  1339118686),
         INT32_C(  1982441324), -INT32_C(   635320481), -INT32_C(  1088769450),  INT32_C(  1675067948),  INT32_C(   261564386),  INT32_C(    39659857),  INT32_C(   586321280), -INT32_C(  1529732808) },
      {  INT32_C(   656145352),  INT32_C(   402734274),  INT32_C(   450370798), -INT32_C(  2139181154),  INT32_C(   663754198), -INT32_C(  1104548546), -INT32_C(  1277158278),  INT32_C(    89633341),
         INT32_C(  1865183917), -INT32_C(  1668797010), -INT32_C(   407412663), -INT32_C(   446155505), -INT32_C(  2012350390),  INT32_C(  1615214309), -INT32_C(  1894570414), -INT32_C(  2053870888) },
      { -INT32_C(  1438818527),  INT32_C(    84271486), -INT32_C(   897299084),  INT32_C(    65722796), -INT32_C(   774654206), -INT32_C(  1596765268),  INT32_C(  1722704944), -INT32_C(  1702445486),
         INT32_C(   977610126),  INT32_C(   624880216), -INT32_C(   420412203),  INT32_C(   669664809),  INT32_C(   972684653),  INT32_C(  1591338821), -INT32_C(   507131901),  INT32_C(  1048291525) } },
    { { -INT32_C(  1946894115),  INT32_C(   925400302), -INT32_C(   350232612),  INT32_C(  1590789908),  INT32_C(  1692851839),  INT32_C(  1740909588),  INT32_C(   720820050),  INT32_C(   531598146),
         INT32_C(   967484235), -INT32_C(    59649504), -INT32_C(   974614351), -INT32_C(  1776043753), -INT32_C(  1409676905), -INT32_C(  2028814539), -INT32_C(   659486314), -INT32_C(   537436012) },
      {  INT32_C(   605659652),  INT32_C(   639666804),  INT32_C(   837486618),  INT32_C(  1489440705),  INT32_C(  1308934424),  INT32_C(   399840896), -INT32_C(  1276147937), -INT32_C(   342628377),
        -INT32_C(    49304439),  INT32_C(  1344483382), -INT32_C(    92205256),  INT32_C(   911362078), -INT32_C(  1971038711), -INT32_C(  1935582611), -INT32_C(   935292703),  INT32_C(    11850615) },
      { -INT32_C(  1241595009),  INT32_C(   738533875),  INT32_C(  1311147568), -INT32_C(   679118642),  INT32_C(  1012992463),  INT32_C(  1153958499),  INT32_C(   168626323),  INT32_C(  1510719963),
         INT32_C(  2014251396),  INT32_C(  1537479722),  INT32_C(  1755957914),  INT32_C(   289418818), -INT32_C(  1689411272),  INT32_C(   937432740), -INT32_C(    96342754),  INT32_C(   844385454) },
      { -INT32_C(  1946906239),  INT32_C(   925405180), -INT32_C(   350219588),  INT32_C(  1590753973),  INT32_C(  1692857468),  INT32_C(  1740911940),  INT32_C(   720822648),  INT32_C(   531597813),
         INT32_C(   967499380), -INT32_C(    59642216), -INT32_C(   974606096), -INT32_C(  1776032295), -INT32_C(  1409688345), -INT32_C(  2028820039), -INT32_C(   659479340), -INT32_C(   537414614) } },
    { { -INT32_C(  2136316843),  INT32_C(   333139576),  INT32_C(  1534821400), -INT32_C(   345195597),  INT32_C(    25606749), -INT32_C(   298293552),  INT32_C(    32012627),  INT32_C(   456408518),
         INT32_C(   429645473),  INT32_C(  1160541741), -INT32_C(  1365202693), -INT32_C(  1063711389), -INT32_C(  1748951097), -INT32_C(   645531258),  INT32_C(   970681971),  INT32_C(  1280577451) },
      {  INT32_C(   442888429),  INT32_C(  1633718886), -INT32_C(  1659961286), -INT32_C(   731994099),  INT32_C(  1315643336), -INT32_C(  1943539431),  INT32_C(   180683359), -INT32_C(    11134703),
         INT32_C(  1880734473), -INT32_C(  2016315059), -INT32_C(  2027560582),  INT32_C(  1365017481), -INT32_C(  1147156574),  INT32_C(   390580152), -INT32_C(   618591031),  INT32_C(   802846502) },
      { -INT32_C(  2137001165), -INT32_C(   418877075), -INT32_C(   613536430),  INT32_C(  1378732720),  INT32_C(  1242483858),  INT32_C(  1566725523), -INT32_C(  2009562270),  INT32_C(   783749883),
         INT32_C(  1924028165),  INT32_C(   442087112), -INT32_C(  1812608797),  INT32_C(   619061906), -INT32_C(  2106592017), -INT32_C(  1411395255),  INT32_C(  1328748372),  INT32_C(   796781353) },
      { -INT32_C(  2136321001),  INT32_C(   333165535),  INT32_C(  1534821997), -INT32_C(   345184095),  INT32_C(    25590407), -INT32_C(   298258950),  INT32_C(    32031519),  INT32_C(   456414335),
         INT32_C(   429672505),  INT32_C(  1160550586), -INT32_C(  1365233953), -INT32_C(  1063721481), -INT32_C(  1748962152), -INT32_C(   645531477),  INT32_C(   970718115),  INT32_C(  1280605969) } },
    { {  INT32_C(   178334786), -INT32_C(   970654750),  INT32_C(  1431902659),  INT32_C(   729431868),  INT32_C(  2108549427),  INT32_C(   237538746), -INT32_C(   832676700), -INT32_C(  1979851961),
        -INT32_C(   359424505),  INT32_C(  1555085209),  INT32_C(   212994512),  INT32_C(  2083990601), -INT32_C(   805706475),  INT32_C(   383591026), -INT32_C(   974898306),  INT32_C(   508485911) },
      {  INT32_C(   420012416),  INT32_C(  1752610968),  INT32_C(   192227522),  INT32_C(  1770499156),  INT32_C(    54034833),  INT32_C(   555357603),  INT32_C(  1759968849),  INT32_C(  1619408096),
        -INT32_C(  1384542443),  INT32_C(   152432455),  INT32_C(  1796508183), -INT32_C(   925590473), -INT32_C(  1060369379),  INT32_C(  1944184354), -INT32_C(   992229404),  INT32_C(   270819835) },
      {  INT32_C(   951951088), -INT32_C(  1522412915), -INT32_C(  1810868643),  INT32_C(   257746418),  INT32_C(   332343537), -INT32_C(   209276914),  INT32_C(  1924620663), -INT32_C(  1283269437),
         INT32_C(   132857722),  INT32_C(  1856777489),  INT32_C(  1963113859), -INT32_C(  1820041310), -INT32_C(  1767419001),  INT32_C(  2055810307),  INT32_C(  1391214735), -INT32_C(  1778029028) },
      {  INT32_C(   178311552), -INT32_C(   970682488),  INT32_C(  1431924825),  INT32_C(   729440043),  INT32_C(  2108549725),  INT32_C(   237535747), -INT32_C(   832647357), -INT32_C(  1979891773),
        -INT32_C(   359414264),  INT32_C(  1555096397),  INT32_C(   212994950),  INT32_C(  2083952163), -INT32_C(   805747496),  INT32_C(   383588733), -INT32_C(   974899642),  INT32_C(   508502093) } },
    { { -INT32_C(  1096879699), -INT32_C(  1590867426), -INT32_C(  1458163961),  INT32_C(   373136014), -INT32_C(   257104659), -INT32_C(  1603652335), -INT32_C(  1829611915),  INT32_C(  1898510532),
         INT32_C(   120571625),  INT32_C(   413686801),  INT32_C(   448970380),  INT32_C(  1160839000), -INT32_C(   197796637), -INT32_C(  2020237551), -INT32_C(  1189509131),  INT32_C(  1747599743) },
      {  INT32_C(   426727688),  INT32_C(  1110513590),  INT32_C(   761066453), -INT32_C(   713912846),  INT32_C(  2076878697),  INT32_C(   990011206),  INT32_C(  1727273958),  INT32_C(  1691229788),
         INT32_C(   779959928),  INT32_C(   728805205), -INT32_C(  1789342558), -INT32_C(  1016411303), -INT32_C(  1187105678),  INT32_C(  2062827667), -INT32_C(  1209996965),  INT32_C(  2132585991) },
      {  INT32_C(  1118673388), -INT32_C(   345170616),  INT32_C(  1132512746),  INT32_C(    50784912), -INT32_C(  1296284641), -INT32_C(   550719356), -INT32_C(  1583936359), -INT32_C(  1491029061),
        -INT32_C(  1779839412), -INT32_C(   713009429), -INT32_C(  1407647716),  INT32_C(   162471914), -INT32_C(   390304924), -INT32_C(  1245124580), -INT32_C(  1336516619),  INT32_C(  1582790418) },
      { -INT32_C(  1096896589), -INT32_C(  1590849700), -INT32_C(  1458191745),  INT32_C(   373107131), -INT32_C(   257113378), -INT32_C(  1603670474), -INT32_C(  1829670591),  INT32_C(  1898499566),
         INT32_C(   120569848),  INT32_C(   413683881),  INT32_C(   448964600),  INT32_C(  1160836504), -INT32_C(   197788329), -INT32_C(  2020257785), -INT32_C(  1189483373),  INT32_C(  1747634949) } },
    { {  INT32_C(   821248325), -INT32_C(  1291422825),  INT32_C(  1583357811), -INT32_C(  1570304194), -INT32_C(  1752489093),  INT32_C(    21779212), -INT32_C(   994991182),  INT32_C(  1596065818),
        -INT32_C(   510716343), -INT32_C(    40594039), -INT32_C(   212077388),  INT32_C(  2140520964), -INT32_C(   233430810),  INT32_C(   636707443),  INT32_C(   535405573), -INT32_C(   159511380) },
      { -INT32_C(  1411969502),  INT32_C(  1487432611),  INT32_C(  1682637664), -INT32_C(  1394351930),  INT32_C(  1956575489),  INT32_C(  1637519707), -INT32_C(   511671499), -INT32_C(  1311244401),
        -INT32_C(  1336037875),  INT32_C(  2047345946), -INT32_C(   807513335),  INT32_C(   897302836),  INT32_C(   363403706), -INT32_C(   545897558),  INT32_C(  1455486919),  INT32_C(    50895094) },
      {  INT32_C(  1622369606),  INT32_C(  1943714922),  INT32_C(  1128445967),  INT32_C(   863550841), -INT32_C(  2142690602),  INT32_C(   744472421), -INT32_C(  1400692554), -INT32_C(    22049864),
         INT32_C(  1516200944),  INT32_C(   785266975),  INT32_C(  1802637554), -INT32_C(  1533023538),  INT32_C(  1915086860),  INT32_C(  1587446951),  INT32_C(  1544167844), -INT32_C(  1671710036) },
      {  INT32_C(   821251980), -INT32_C(  1291409087),  INT32_C(  1583370685), -INT32_C(  1570259232), -INT32_C(  1752483983),  INT32_C(    21797876), -INT32_C(   995025812),  INT32_C(  1596007918),
        -INT32_C(   510674650), -INT32_C(    40587736), -INT32_C(   212028729),  INT32_C(  2140497311), -INT32_C(   233420531),  INT32_C(   636693403),  INT32_C(   535405248), -INT32_C(   159542256) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi32(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r = easysimd_mm512_dpbusd_epi32(src, a, b);
    easysimd_test_x86_assert_equal_i32x16(r, easysimd_mm512_loadu_epi32(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    easysimd__m512i src = easysimd_test_x86_random_i32x16();
    easysimd__m512i a = easysimd_test_x86_random_i32x16();
    easysimd__m512i b = easysimd_test_x86_random_i32x16();
    easysimd__m512i r = easysimd_mm512_dpbusd_epi32(src, a, b);

    easysimd_test_x86_write_i32x16(2, src, EASYSIMD_TEST_VEC_POS_FIRST);
    easysimd_test_x86_write_i32x16(2, a, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, b, EASYSIMD_TEST_VEC_POS_MIDDLE);
    easysimd_test_x86_write_i32x16(2, r, EASYSIMD_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_easysimd_mm512_mask_dpbusd_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t src[16];
    const easysimd__mmask16 k;
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { {  INT32_C(  1881149460),  INT32_C(  1871338058),  INT32_C(  1101599043), -INT32_C(  1256270630),  INT32_C(  1038398141),  INT32_C(  1977646053), -INT32_C(   770472418),  INT32_C(   953817124),
         INT32_C(  1923676712), -INT32_C(  1746849196),  INT32_C(   517507907),  INT32_C(   483653215), -INT32_C(  1923434584),  INT32_C(  1342323250), -INT32_C(   450751040),  INT32_C(  1444805678) },
      UINT16_C(50934),
      { -INT32_C(  1426568247), -INT32_C(  1154138910),  INT32_C(   766612826),  INT32_C(   199514801), -INT32_C(   364505369), -INT32_C(  1996552602),  INT32_C(   159657451),  INT32_C(  1305442948),
        -INT32_C(  1460156475),  INT32_C(  1566780675),  INT32_C(  1938429122),  INT32_C(  1451191918), -INT32_C(   364853884), -INT32_C(  1217249333), -INT32_C(   239012243),  INT32_C(   910069617) },
      {  INT32_C(  1524512086),  INT32_C(   615989602), -INT32_C(   996654506),  INT32_C(   890902448), -INT32_C(  1491117348),  INT32_C(   123638426), -INT32_C(   101179768),  INT32_C(    70268589),
        -INT32_C(   832696725), -INT32_C(  1494084272),  INT32_C(   141199959),  INT32_C(  2101183905),  INT32_C(  2032491743),  INT32_C(  2004911086),  INT32_C(  1332770978),  INT32_C(   441688238) },
      {  INT32_C(  1881149460),  INT32_C(  1871366969),  INT32_C(  1101595509), -INT32_C(  1256270630),  INT32_C(  1038373149),  INT32_C(  1977659911), -INT32_C(   770500387),  INT32_C(   953823000),
         INT32_C(  1923676712), -INT32_C(  1746857767),  INT32_C(   517537973),  INT32_C(   483653215), -INT32_C(  1923434584),  INT32_C(  1342323250), -INT32_C(   450691223),  INT32_C(  1444789234) } },
    { { -INT32_C(     1527377),  INT32_C(   514185927),  INT32_C(   103157605),  INT32_C(  1954833300), -INT32_C(  1360156224), -INT32_C(   836407764), -INT32_C(  1793157402), -INT32_C(   441487050),
        -INT32_C(   354117853), -INT32_C(   704083599),  INT32_C(   769470361),  INT32_C(  1386307986),  INT32_C(   906071818), -INT32_C(   486201603), -INT32_C(   210230340), -INT32_C(  1210570860) },
      UINT16_C(48574),
      { -INT32_C(  1438240607), -INT32_C(   472260858), -INT32_C(  1371247859), -INT32_C(  1086501186), -INT32_C(  1964688763), -INT32_C(  1767005666),  INT32_C(  1841185173), -INT32_C(  1708491527),
        -INT32_C(  1287360084),  INT32_C(  1570118992),  INT32_C(  1192024969), -INT32_C(  1375319767), -INT32_C(  1573327996),  INT32_C(   574154125),  INT32_C(   546371111),  INT32_C(   532331123) },
      {  INT32_C(  2094202667), -INT32_C(  1512478436),  INT32_C(  1844241732), -INT32_C(  1289948625),  INT32_C(  1800754398),  INT32_C(  1636732218), -INT32_C(   142532988),  INT32_C(    68565976),
         INT32_C(  1468066106), -INT32_C(  1761846958),  INT32_C(  1845750079), -INT32_C(  1189011493), -INT32_C(  1373276556), -INT32_C(  2029014269), -INT32_C(  1451323183),  INT32_C(   112039371) },
      { -INT32_C(     1527377),  INT32_C(   514180390),  INT32_C(   103173206),  INT32_C(  1954828139), -INT32_C(  1360121643), -INT32_C(   836429826), -INT32_C(  1793157402), -INT32_C(   441488154),
        -INT32_C(   354103607), -INT32_C(   704083599),  INT32_C(   769481801),  INT32_C(  1386296576),  INT32_C(   906103648), -INT32_C(   486222087), -INT32_C(   210230340), -INT32_C(  1210612109) } },
    { { -INT32_C(   799199874), -INT32_C(   966370937),  INT32_C(   506751298), -INT32_C(    36219511), -INT32_C(   827589429), -INT32_C(  2141799761),  INT32_C(   355128394), -INT32_C(   417605783),
        -INT32_C(  1934133243),  INT32_C(   340925906),  INT32_C(   271746695), -INT32_C(  1509094693), -INT32_C(  1233864698), -INT32_C(  1137259918),  INT32_C(   147939487),  INT32_C(  1039199544) },
      UINT16_C(42853),
      {  INT32_C(   465909705),  INT32_C(  2124500044),  INT32_C(  1770552412), -INT32_C(  1776185566),  INT32_C(  2069992261),  INT32_C(   551288911), -INT32_C(   133360632),  INT32_C(   446722897),
        -INT32_C(   164272726),  INT32_C(   225826481),  INT32_C(  1953955154), -INT32_C(   787834996),  INT32_C(  2035116842),  INT32_C(  1956194667), -INT32_C(  1905481923), -INT32_C(   995619814) },
      {  INT32_C(   582737265),  INT32_C(    86978739), -INT32_C(  1166432979),  INT32_C(  1720419132),  INT32_C(  1524619503),  INT32_C(  1053718785),  INT32_C(   986463008), -INT32_C(  1174506425),
         INT32_C(    81508689),  INT32_C(   386468841), -INT32_C(   305036624), -INT32_C(   179086330),  INT32_C(   911160117), -INT32_C(   864739668), -INT32_C(  1610202791),  INT32_C(   123274422) },
      { -INT32_C(   799191761), -INT32_C(   966370937),  INT32_C(   506753260), -INT32_C(    36219511), -INT32_C(   827589429), -INT32_C(  2141808698),  INT32_C(   355143538), -INT32_C(   417605783),
        -INT32_C(  1934127621),  INT32_C(   340925541),  INT32_C(   271700507), -INT32_C(  1509094693), -INT32_C(  1233864698), -INT32_C(  1137255807),  INT32_C(   147939487),  INT32_C(  1039213992) } },
    { { -INT32_C(  1492372034), -INT32_C(   255978176), -INT32_C(  1629646952),  INT32_C(   563294700),  INT32_C(   274195044),  INT32_C(  1507642368), -INT32_C(   990191090), -INT32_C(  1530178586),
        -INT32_C(   934488184), -INT32_C(  2051470611), -INT32_C(  2044488038),  INT32_C(   732411591), -INT32_C(  1724121448), -INT32_C(   638445621), -INT32_C(   526521095), -INT32_C(   930846656) },
      UINT16_C(53312),
      {  INT32_C(  1255878033), -INT32_C(   706710094), -INT32_C(  1567905541),  INT32_C(   262153171), -INT32_C(  1356436548), -INT32_C(   509927356), -INT32_C(  2075469056),  INT32_C(   894798500),
         INT32_C(  1769943223), -INT32_C(  1606525019), -INT32_C(   616380153), -INT32_C(  1444224276), -INT32_C(  1839722418),  INT32_C(   812970800),  INT32_C(  1941290703), -INT32_C(     5764536) },
      { -INT32_C(   546822342), -INT32_C(  1904236922),  INT32_C(  1533657455), -INT32_C(   251374685), -INT32_C(  1803264925),  INT32_C(   533002320), -INT32_C(    23955018), -INT32_C(  1107477885),
        -INT32_C(   409180831),  INT32_C(  2054494987), -INT32_C(  2133467428), -INT32_C(  1787700687), -INT32_C(  2044070602), -INT32_C(  1549406739), -INT32_C(   358533273), -INT32_C(   760701327) },
      { -INT32_C(  1492372034), -INT32_C(   255978176), -INT32_C(  1629646952),  INT32_C(   563294700),  INT32_C(   274195044),  INT32_C(  1507642368), -INT32_C(   990172995), -INT32_C(  1530178586),
        -INT32_C(   934488184), -INT32_C(  2051470611), -INT32_C(  2044488038),  INT32_C(   732411591), -INT32_C(  1724131616), -INT32_C(   638445621), -INT32_C(   526509049), -INT32_C(   930866014) } },
    { {  INT32_C(   247088131),  INT32_C(  1015557984),  INT32_C(  1069309454),  INT32_C(  1859399224),  INT32_C(   301333795),  INT32_C(  1387567851),  INT32_C(  1128027858), -INT32_C(   132717324),
        -INT32_C(  1996042199),  INT32_C(   231051263),  INT32_C(   625836781), -INT32_C(   745332304),  INT32_C(   182749215), -INT32_C(   178480861), -INT32_C(   482830097), -INT32_C(  1495576963) },
      UINT16_C(57886),
      { -INT32_C(   193913297),  INT32_C(  2021023275),  INT32_C(   379201155), -INT32_C(   559957510),  INT32_C(   527942339), -INT32_C(   273193289),  INT32_C(   624833610),  INT32_C(   151477466),
        -INT32_C(  1493272454),  INT32_C(  1495168214), -INT32_C(  1787774821),  INT32_C(   879955825), -INT32_C(  2007766063), -INT32_C(  1686697135),  INT32_C(   448902463), -INT32_C(  1943812078) },
      {  INT32_C(   389161281),  INT32_C(   829510038),  INT32_C(  2076631305), -INT32_C(  1045480976),  INT32_C(  1984496420),  INT32_C(  1293008910), -INT32_C(  2006461834), -INT32_C(   602633317),
         INT32_C(  1123239852), -INT32_C(  1586273128),  INT32_C(   891042117), -INT32_C(  1745433485), -INT32_C(   586334257),  INT32_C(  1982471936), -INT32_C(  1929473295), -INT32_C(   915926499) },
      {  INT32_C(   247088131),  INT32_C(  1015580254),  INT32_C(  1069303229),  INT32_C(  1859378904),  INT32_C(   301353742),  INT32_C(  1387567851),  INT32_C(  1128027858), -INT32_C(   132717324),
        -INT32_C(  1996042199),  INT32_C(   231035602),  INT32_C(   625836781), -INT32_C(   745332304),  INT32_C(   182749215), -INT32_C(   178457232), -INT32_C(   482854354), -INT32_C(  1495576901) } },
    { { -INT32_C(   250848167),  INT32_C(   110329792),  INT32_C(   742109113),  INT32_C(  1254306427),  INT32_C(  1898434929), -INT32_C(   504933648),  INT32_C(    24045028),  INT32_C(  1372247800),
        -INT32_C(   213658062),  INT32_C(   268031574),  INT32_C(    20657285), -INT32_C(   666108314), -INT32_C(  1085705265), -INT32_C(  1449053755),  INT32_C(   246026006),  INT32_C(   408974565) },
      UINT16_C(41803),
      {  INT32_C(    75079947), -INT32_C(   348586320),  INT32_C(  1273602047), -INT32_C(  1061242505),  INT32_C(   435258232), -INT32_C(   702019540),  INT32_C(  1984564758),  INT32_C(   823760166),
        -INT32_C(   432696778), -INT32_C(  1865323119), -INT32_C(  2082751732), -INT32_C(   331048588),  INT32_C(  1208366364),  INT32_C(  1411264061),  INT32_C(  1657432380),  INT32_C(   898884862) },
      {  INT32_C(   119261302),  INT32_C(  1117318454),  INT32_C(   482767784),  INT32_C(   688392717),  INT32_C(  2104626751),  INT32_C(  2043777085), -INT32_C(   119759879), -INT32_C(   164794497),
         INT32_C(  1862092856), -INT32_C(   575630027),  INT32_C(   385447433), -INT32_C(  1069612416),  INT32_C(  1295888656),  INT32_C(  1002901058),  INT32_C(   691250089),  INT32_C(  1243570194) },
      { -INT32_C(   250852590),  INT32_C(   110344156),  INT32_C(   742109113),  INT32_C(  1254319216),  INT32_C(  1898434929), -INT32_C(   504933648),  INT32_C(    24039852),  INT32_C(  1372247800),
        -INT32_C(   213619385),  INT32_C(   268005836),  INT32_C(    20657285), -INT32_C(   666108314), -INT32_C(  1085705265), -INT32_C(  1449045839),  INT32_C(   246026006),  INT32_C(   409009504) } },
    { { -INT32_C(   558359383), -INT32_C(  1145280078),  INT32_C(  1624356319), -INT32_C(   937422665),  INT32_C(    68509122),  INT32_C(   339729515), -INT32_C(  1841466497),  INT32_C(  2094816467),
         INT32_C(   727422329), -INT32_C(   572123138), -INT32_C(  2076330036), -INT32_C(  1991483961),  INT32_C(   630022586), -INT32_C(  1120219842),  INT32_C(   323974976),  INT32_C(  1301294292) },
      UINT16_C(60353),
      {  INT32_C(  1577238392), -INT32_C(   636105060), -INT32_C(  1640506286), -INT32_C(   218041754),  INT32_C(  1371553303),  INT32_C(  1271464187), -INT32_C(  1535664876),  INT32_C(  1653553386),
        -INT32_C(  1816096265), -INT32_C(  1301424801),  INT32_C(   391161265), -INT32_C(  1374990185), -INT32_C(  1979659378), -INT32_C(   522860084),  INT32_C(  1350847590),  INT32_C(  2075267972) },
      {  INT32_C(    51278500), -INT32_C(   122323897), -INT32_C(  1206909407), -INT32_C(   446293162), -INT32_C(  1301322010), -INT32_C(  1785576401),  INT32_C(   350557840), -INT32_C(   846227671),
         INT32_C(  1355914505),  INT32_C(   977896985), -INT32_C(   470656628),  INT32_C(  1523079540), -INT32_C(   284346433),  INT32_C(   193240955), -INT32_C(   551589194),  INT32_C(   195931649) },
      { -INT32_C(   558348339), -INT32_C(  1145280078),  INT32_C(  1624356319), -INT32_C(   937422665),  INT32_C(    68509122),  INT32_C(   339729515), -INT32_C(  1841465216),  INT32_C(  2094799024),
         INT32_C(   727412933), -INT32_C(   572128468), -INT32_C(  2076330036), -INT32_C(  1991444281),  INT32_C(   630022586), -INT32_C(  1120238195),  INT32_C(   323976936),  INT32_C(  1301279445) } },
    { {  INT32_C(  1700494923), -INT32_C(  1851808764),  INT32_C(  1903465213), -INT32_C(  1429455637), -INT32_C(   291907213),  INT32_C(   788078200), -INT32_C(  1995564920),  INT32_C(   294960070),
         INT32_C(  1031204921), -INT32_C(  1831987564), -INT32_C(  1828502872), -INT32_C(   247607426),  INT32_C(   568317864),  INT32_C(  2102384885), -INT32_C(  1241096720),  INT32_C(  1372101400) },
      UINT16_C(16011),
      {  INT32_C(  1565794191), -INT32_C(  1247740751), -INT32_C(   846979441), -INT32_C(   274453232), -INT32_C(  1647797938),  INT32_C(   486258710),  INT32_C(   934744943), -INT32_C(   193641883),
         INT32_C(   324127330),  INT32_C(  1456009670),  INT32_C(   539184400), -INT32_C(   938489990),  INT32_C(  1986385760), -INT32_C(     7184240), -INT32_C(   650753420), -INT32_C(   288511092) },
      {  INT32_C(  1006771829),  INT32_C(   546490896), -INT32_C(  1858030313), -INT32_C(   598126724), -INT32_C(  1219313881), -INT32_C(  1816730593), -INT32_C(  1150489554),  INT32_C(   229194135),
         INT32_C(  1749658456), -INT32_C(  1920410762),  INT32_C(   203409552),  INT32_C(  1055422487),  INT32_C(  1475689015),  INT32_C(  1307224862),  INT32_C(   789075863), -INT32_C(   398675568) },
      {  INT32_C(  1700518332), -INT32_C(  1851831348),  INT32_C(  1903465213), -INT32_C(  1429444106), -INT32_C(   291907213),  INT32_C(   788078200), -INT32_C(  1995564920),  INT32_C(   294946220),
         INT32_C(  1031204921), -INT32_C(  1832007007), -INT32_C(  1828507507), -INT32_C(   247568700),  INT32_C(   568344769),  INT32_C(  2102397468), -INT32_C(  1241096720),  INT32_C(  1372101400) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi32(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r = easysimd_mm512_mask_dpbusd_epi32(src, test_vec[i].k, a, b);
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
    easysimd__m512i r = easysimd_mm512_mask_dpbusd_epi32(src, k, a, b);

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
test_easysimd_mm512_maskz_dpbusd_epi32 (EASYSIMD_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const easysimd__mmask16 k;
    const int32_t src[16];
    const int32_t a[16];
    const int32_t b[16];
    const int32_t r[16];
  } test_vec[] = {
    { UINT16_C(34141),
      { -INT32_C(   664743088),  INT32_C(  2141253728), -INT32_C(   453461764), -INT32_C(   333565962),  INT32_C(  1905736838), -INT32_C(  1849151607), -INT32_C(  1706862498), -INT32_C(  1860198335),
        -INT32_C(   747995277),  INT32_C(  1800604271), -INT32_C(  1185985598),  INT32_C(    27618682),  INT32_C(   863124649), -INT32_C(   909886869), -INT32_C(   748484718),  INT32_C(   442860199) },
      {  INT32_C(  1894698753), -INT32_C(  1663352359),  INT32_C(   106244748),  INT32_C(  1091041943), -INT32_C(  1569425098),  INT32_C(  1181431987), -INT32_C(   417738944),  INT32_C(  1392672337),
         INT32_C(   667152461), -INT32_C(  1111253199),  INT32_C(  1640175817),  INT32_C(  1218628370), -INT32_C(   118876604), -INT32_C(  1908517298),  INT32_C(  1987467045),  INT32_C(   600406230) },
      { -INT32_C(  1723167384), -INT32_C(   178909908),  INT32_C(   928389413),  INT32_C(   696318180),  INT32_C(  1562470926), -INT32_C(   437559360), -INT32_C(  1940102730),  INT32_C(  1102063065),
        -INT32_C(   556074318),  INT32_C(   768880648),  INT32_C(   778381898),  INT32_C(   827843875),  INT32_C(   277772367), -INT32_C(  1896514857), -INT32_C(  1256566309),  INT32_C(   704039543) },
      { -INT32_C(   664760713),  INT32_C(           0), -INT32_C(   453447894), -INT32_C(   333570421),  INT32_C(  1905769314),  INT32_C(           0), -INT32_C(  1706871651),  INT32_C(           0),
        -INT32_C(   748011459),  INT32_C(           0), -INT32_C(  1185945559),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(   442878610) } },
    { UINT16_C(53701),
      { -INT32_C(   603861752),  INT32_C(  1594248186), -INT32_C(   784062086), -INT32_C(   380988325), -INT32_C(  1721556572), -INT32_C(   890552401), -INT32_C(   359374092), -INT32_C(  1816438389),
         INT32_C(   560971046), -INT32_C(  2105510392), -INT32_C(    95107681), -INT32_C(    35348903),  INT32_C(  1872119743),  INT32_C(  2050589062),  INT32_C(  1868942819),  INT32_C(  1291984935) },
      { -INT32_C(   428969251), -INT32_C(  2023165976),  INT32_C(   226671796),  INT32_C(   436889178),  INT32_C(   881434797),  INT32_C(    95339042), -INT32_C(  1233906801),  INT32_C(   285505332),
        -INT32_C(   772312343),  INT32_C(   358113377),  INT32_C(  1998772764), -INT32_C(   292475840), -INT32_C(   282977587),  INT32_C(  1811206364),  INT32_C(   404908516), -INT32_C(   920049952) },
      { -INT32_C(    90562152), -INT32_C(  1659899263),  INT32_C(   219427533),  INT32_C(   737912158), -INT32_C(  1692787265), -INT32_C(   771289106),  INT32_C(  1491741048), -INT32_C(   417262769),
        -INT32_C(  1243497676),  INT32_C(  2052321709), -INT32_C(  2121767133), -INT32_C(   861043955), -INT32_C(  1888958559),  INT32_C(  1315008470), -INT32_C(   408532072), -INT32_C(  1815165090) },
      { -INT32_C(   603893574),  INT32_C(           0), -INT32_C(   784059097),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   359342689), -INT32_C(  1816432313),
         INT32_C(   560951964),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1872075315),  INT32_C(           0),  INT32_C(  1868923322),  INT32_C(  1291979866) } },
    { UINT16_C(45186),
      { -INT32_C(  1683935160),  INT32_C(   839107754), -INT32_C(   222949307),  INT32_C(  1153062876),  INT32_C(  1202950374), -INT32_C(  2054009889),  INT32_C(    21884978), -INT32_C(   860762237),
        -INT32_C(  1436069121),  INT32_C(  1541171734),  INT32_C(  1464767098), -INT32_C(   811923223),  INT32_C(  1997950872), -INT32_C(   839014246),  INT32_C(   483281561),  INT32_C(   434667289) },
      { -INT32_C(   406630191),  INT32_C(   893558714),  INT32_C(   462196786),  INT32_C(   837494680),  INT32_C(   296223094),  INT32_C(  1138664874),  INT32_C(   157265135), -INT32_C(    64862165),
         INT32_C(  1390667160), -INT32_C(  1232657020),  INT32_C(  1322390454), -INT32_C(  1317028549),  INT32_C(  1757554878), -INT32_C(  1112825651),  INT32_C(  2009467724), -INT32_C(   344725421) },
      {  INT32_C(  1362974413),  INT32_C(   839435644),  INT32_C(   327211736),  INT32_C(  1438974103), -INT32_C(   172062936),  INT32_C(  1957849384), -INT32_C(   940803980),  INT32_C(   766664544),
         INT32_C(   847245494), -INT32_C(  1922791499), -INT32_C(   123607967),  INT32_C(   206399204),  INT32_C(   352455661), -INT32_C(   376785803), -INT32_C(  1917749715), -INT32_C(  1950653483) },
      {  INT32_C(           0),  INT32_C(   839124615),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   860742581),
         INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1997950274), -INT32_C(   839026946),  INT32_C(           0),  INT32_C(   434651258) } },
    { UINT16_C(14931),
      {  INT32_C(   566298813),  INT32_C(   923148950),  INT32_C(  1738402330), -INT32_C(   109868297),  INT32_C(   716040352), -INT32_C(  2103387439),  INT32_C(   602305896), -INT32_C(  1101186815),
        -INT32_C(   656400830),  INT32_C(  1510991424), -INT32_C(   943608624), -INT32_C(   692046794), -INT32_C(   301961700), -INT32_C(  1334796216),  INT32_C(   399726102), -INT32_C(   757780336) },
      { -INT32_C(  1901415090),  INT32_C(  1827191195), -INT32_C(  1691112859), -INT32_C(    76352290), -INT32_C(  1427541406),  INT32_C(   693852435),  INT32_C(  1077948080), -INT32_C(  1391323809),
         INT32_C(  1715256523), -INT32_C(   623762315), -INT32_C(  1384839474),  INT32_C(  1554573306),  INT32_C(  1829146970), -INT32_C(  1701420566), -INT32_C(   270870896), -INT32_C(  1231229717) },
      {  INT32_C(   488495272), -INT32_C(   872943619), -INT32_C(   277320203), -INT32_C(  1370808236), -INT32_C(  1675930959),  INT32_C(  1127657907),  INT32_C(  1932661127), -INT32_C(  1507209219),
        -INT32_C(  1530706265),  INT32_C(   728808246),  INT32_C(  2082203688), -INT32_C(  1171560951),  INT32_C(  1817593528),  INT32_C(  2142211576), -INT32_C(  1678581090),  INT32_C(  1480662193) },
      {  INT32_C(   566293757),  INT32_C(   923137528),  INT32_C(           0),  INT32_C(           0),  INT32_C(   716031249),  INT32_C(           0),  INT32_C(   602295942),  INT32_C(           0),
         INT32_C(           0),  INT32_C(  1511028152),  INT32_C(           0), -INT32_C(   692020198), -INT32_C(   301945742), -INT32_C(  1334801835),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C( 1378),
      {  INT32_C(  1841338621), -INT32_C(   548017980), -INT32_C(  1891279260),  INT32_C(  1876360729),  INT32_C(   435998314),  INT32_C(  1090230861),  INT32_C(  2002562102),  INT32_C(    41729541),
         INT32_C(   460274775), -INT32_C(  1980054492),  INT32_C(  1008221987), -INT32_C(  1481904579),  INT32_C(   180463804),  INT32_C(  2018163778),  INT32_C(  1861265001), -INT32_C(  1150260124) },
      { -INT32_C(   841556055), -INT32_C(   967389021),  INT32_C(  1292005136),  INT32_C(   452308573), -INT32_C(  1742424490), -INT32_C(   619614606),  INT32_C(  2018050324),  INT32_C(   372554093),
         INT32_C(  1021577880), -INT32_C(   352175397),  INT32_C(   121111977),  INT32_C(   153169331),  INT32_C(  1453409763), -INT32_C(   953044301),  INT32_C(   557808563), -INT32_C(   852004044) },
      {  INT32_C(  1493769086), -INT32_C(    12317866), -INT32_C(  1022985200), -INT32_C(  1916000342),  INT32_C(   534998636), -INT32_C(   723118816), -INT32_C(   990566768),  INT32_C(   395390105),
        -INT32_C(  1636787640), -INT32_C(  1231178586), -INT32_C(   612719567),  INT32_C(   946357963), -INT32_C(   715699020), -INT32_C(   240566687), -INT32_C(    38429084),  INT32_C(   303384522) },
      {  INT32_C(           0), -INT32_C(   547996024),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(  1090226741),  INT32_C(  2002551897),  INT32_C(           0),
         INT32_C(   460304355),  INT32_C(           0),  INT32_C(  1008236381),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0) } },
    { UINT16_C(34273),
      {  INT32_C(  1312458672), -INT32_C(  1192072386), -INT32_C(  1359037114),  INT32_C(  1308275701),  INT32_C(   831216520),  INT32_C(    47181644),  INT32_C(    21600748),  INT32_C(  1552362156),
        -INT32_C(   257244750),  INT32_C(  1923652652),  INT32_C(  1327539802), -INT32_C(   509863079), -INT32_C(  1022220426),  INT32_C(    63299862),  INT32_C(   654577275), -INT32_C(   360477896) },
      {  INT32_C(  2027564620),  INT32_C(   619348682), -INT32_C(  2106324183), -INT32_C(  1671163866),  INT32_C(  1314879032), -INT32_C(   749657000),  INT32_C(  1811568178),  INT32_C(   777354721),
         INT32_C(  1990602923), -INT32_C(   610627150), -INT32_C(  1034023268),  INT32_C(  1465827871), -INT32_C(  1868186056),  INT32_C(   358873058),  INT32_C(   780164429), -INT32_C(  2040736293) },
      { -INT32_C(  1191443707),  INT32_C(   814978964), -INT32_C(   990711387), -INT32_C(   333754189), -INT32_C(   243482354),  INT32_C(    84336824),  INT32_C(   406029885),  INT32_C(  1637847131),
         INT32_C(   655989651), -INT32_C(   665342926),  INT32_C(  1369196958), -INT32_C(  1472350055),  INT32_C(   832158329), -INT32_C(   684286054), -INT32_C(  2131728091), -INT32_C(  1931374599) },
      {  INT32_C(  1312449678),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(    47175697),  INT32_C(    21608624),  INT32_C(  1552364848),
        -INT32_C(   257259485),  INT32_C(           0),  INT32_C(  1327531850),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0),  INT32_C(           0), -INT32_C(   360521894) } },
    { UINT16_C(64042),
      {  INT32_C(   178740659), -INT32_C(   783071947), -INT32_C(   729158250),  INT32_C(   781058709),  INT32_C(  1775052851), -INT32_C(   271322369),  INT32_C(  1417596018),  INT32_C(   206481753),
         INT32_C(   991425798), -INT32_C(   787715526), -INT32_C(   324692393), -INT32_C(   870698088),  INT32_C(  1513482075),  INT32_C(  1279920345),  INT32_C(   782289109),  INT32_C(  2000350833) },
      {  INT32_C(   515002852),  INT32_C(   334479292), -INT32_C(   318729131),  INT32_C(   582556359), -INT32_C(   612503806), -INT32_C(   869808137),  INT32_C(    33212303), -INT32_C(  1703397962),
         INT32_C(  1119366022),  INT32_C(  1062643946),  INT32_C(    70014524),  INT32_C(  1915151984), -INT32_C(   867261483), -INT32_C(    90606230), -INT32_C(   201616579),  INT32_C(  1334670280) },
      { -INT32_C(  2003745378),  INT32_C(   717744109), -INT32_C(  1372654531), -INT32_C(  1356835622),  INT32_C(  1652256504),  INT32_C(   559682788),  INT32_C(  1880381352),  INT32_C(  1774166475),
        -INT32_C(   722316826),  INT32_C(  1996405048), -INT32_C(  2027672403),  INT32_C(  2016822400), -INT32_C(  1747209549),  INT32_C(  1857566662),  INT32_C(  1524616335),  INT32_C(  1405329005) },
      {  INT32_C(           0), -INT32_C(   783093119),  INT32_C(           0),  INT32_C(   781056497),  INT32_C(           0), -INT32_C(   271314985),  INT32_C(           0),  INT32_C(           0),
         INT32_C(           0), -INT32_C(   787707088),  INT32_C(           0), -INT32_C(   870681052),  INT32_C(  1513428654),  INT32_C(  1279937116),  INT32_C(   782288159),  INT32_C(  2000359319) } },
    { UINT16_C(46575),
      {  INT32_C(   628041767), -INT32_C(  1034871650), -INT32_C(   637087068),  INT32_C(   629979466),  INT32_C(   140333904), -INT32_C(  1596658495),  INT32_C(   155140422), -INT32_C(  1128321387),
        -INT32_C(   186503594), -INT32_C(   290049206),  INT32_C(  1355398405), -INT32_C(   948611722),  INT32_C(  1792004776),  INT32_C(    67806398),  INT32_C(  2047756773), -INT32_C(   835203720) },
      {  INT32_C(  1170414075),  INT32_C(  1379170636), -INT32_C(  1381827274), -INT32_C(    76277934), -INT32_C(  1469758486), -INT32_C(   844271641),  INT32_C(   826784697), -INT32_C(  2080407928),
        -INT32_C(   456605033),  INT32_C(  1916206140),  INT32_C(  1277155577), -INT32_C(   649620497), -INT32_C(  1098732329), -INT32_C(   729075941),  INT32_C(  1912984554), -INT32_C(   369818287) },
      {  INT32_C(    63815111), -INT32_C(  1284111430), -INT32_C(   889219621),  INT32_C(    10765865),  INT32_C(   247342834),  INT32_C(  1071794773),  INT32_C(  1857153053), -INT32_C(  1269324051),
         INT32_C(   498607203),  INT32_C(    47263271), -INT32_C(   322055997),  INT32_C(   166490391), -INT32_C(   333993065),  INT32_C(   304872181), -INT32_C(   813638430), -INT32_C(   444344190) },
      {  INT32_C(   628016047), -INT32_C(  1034876785), -INT32_C(   637125641),  INT32_C(   629973766),  INT32_C(           0), -INT32_C(  1596622921),  INT32_C(   155141080), -INT32_C(  1128323082),
        -INT32_C(   186489449),  INT32_C(           0),  INT32_C(  1355369747),  INT32_C(           0),  INT32_C(  1791966771),  INT32_C(    67815612),  INT32_C(           0), -INT32_C(   835251042) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    easysimd__m512i src = easysimd_mm512_loadu_epi32(test_vec[i].src);
    easysimd__m512i a = easysimd_mm512_loadu_epi32(test_vec[i].a);
    easysimd__m512i b = easysimd_mm512_loadu_epi32(test_vec[i].b);
    easysimd__m512i r = easysimd_mm512_maskz_dpbusd_epi32(test_vec[i].k, src, a, b);
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
    easysimd__m512i r = easysimd_mm512_maskz_dpbusd_epi32(k, src, a, b);

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
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_dpbusd_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_mask_dpbusd_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm_maskz_dpbusd_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_dpbusd_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_mask_dpbusd_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm256_maskz_dpbusd_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_dpbusd_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_mask_dpbusd_epi32)
  EASYSIMD_TEST_FUNC_LIST_ENTRY(mm512_maskz_dpbusd_epi32)
EASYSIMD_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
