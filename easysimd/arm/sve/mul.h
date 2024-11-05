#if !defined(EASYSIMD_ARM_SVE_MUL_H)
#define EASYSIMD_ARM_SVE_MUL_H

  #define easysimd_svmul_f32_z(pg, op1, op2)           svmul_f32_z((pg), (op1), (op2))
  #define easysimd_svmul_f64_z(pg, op1, op2)           svmul_f64_z((pg), (op1), (op2))
  #define easysimd_svmul_u32_z(pg, op1, op2)           svmul_u32_z((pg), (op1), (op2))
  #define easysimd_svmul_s16_z(pg, op1, op2)           svmul_s16_z((pg), (op1), (op2))
  #define easysimd_svmul_s32_z(pg, op1, op2)           svmul_s32_z((pg), (op1), (op2))
  #define easysimd_svmul_s64_z(pg, op1, op2)           svmul_s64_z((pg), (op1), (op2))
  #define easysimd_svmulh_s16_z(pg, op1, op2)          svmulh_s16_z((pg), (op1), (op2))
  #define easysimd_svmulh_u16_z(pg, op1, op2)          svmulh_u16_z((pg), (op1), (op2))
  #define easysimd_svmul_n_s32_z(pg, op1, op2)         svmul_n_s32_z((pg), (op1), (op2))
  #define easysimd_svmullb_s32(op1, op2)               svmullb_s32((op1), (op2))
  #define easysimd_svmullb_s64(op1, op2)               svmullb_s64((op1), (op2))
  #define easysimd_svmullb_u32(op1, op2)               svmullb_u32((op1), (op2))
  #define easysimd_svmullb_u64(op1, op2)               svmullb_u64((op1), (op2))

#endif