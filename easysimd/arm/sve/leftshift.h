#if !defined(EASYSIMD_ARM_SVE_LEFTSHIFT_H)
#define EASYSIMD_ARM_SVE_LEFTSHIFT_H

  #define easysimd_svlsl_n_s8_z(pg, op1, op2)            svlsl_n_s8_z((pg), (op1), (op2))
  #define easysimd_svlsl_n_s16_z(pg, op1, op2)           svlsl_n_s16_z((pg), (op1), (op2))
  #define easysimd_svlsl_n_s32_z(pg, op1, op2)           svlsl_n_s32_z((pg), (op1), (op2))
  #define easysimd_svlsl_n_s64_z(pg, op1, op2)           svlsl_n_s64_z((pg), (op1), (op2))
  #define easysimd_svlsl_n_u8_z(pg, op1, op2)            svlsl_n_u8_z((pg), (op1), (op2))
  #define easysimd_svlsl_n_u16_z(pg, op1, op2)           svlsl_n_u16_z((pg), (op1), (op2))
  #define easysimd_svlsl_n_u32_z(pg, op1, op2)           svlsl_n_u32_z((pg), (op1), (op2))
  #define easysimd_svlsl_n_u64_z(pg, op1, op2)           svlsl_n_u64_z((pg), (op1), (op2))

#endif