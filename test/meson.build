cc = meson.get_compiler('c')
libm = cc.find_library('m', required : false)

easysimd_c_defs = ['-DEASYSIMD_CONSTRAINED_COMPILATION']
easysimd_cxx_defs = ['-DEASYSIMD_CONSTRAINED_COMPILATION']
easysimd_c_args = []
easysimd_cxx_args = []
easysimd_deps = [libm]

if get_option('sleef').enabled() or meson.get_external_property('sleef', false)
  sleef = cc.find_library('sleef', required: true)
  easysimd_deps += sleef
  easysimd_c_defs = ['-DEASYSIMD_MATH_SLEEF_ENABLE']
  easysimd_cxx_defs = ['-DEASYSIMD_MATH_SLEEF_ENABLE']
endif

c_openmp_simd = false
cxx_openmp_simd = false
foreach omp_arg : ['-fopenmp-simd', '-qopenmp-simd']
  if (not c_openmp_simd) and cc.has_argument(omp_arg)
    easysimd_c_args += omp_arg
    easysimd_c_defs += '-DEASYSIMD_ENABLE_OPENMP'
    c_openmp_simd = true
  endif

  if (not cxx_openmp_simd) and cxx.has_argument(omp_arg)
    easysimd_cxx_args += omp_arg
    easysimd_cxx_defs += '-DEASYSIMD_ENABLE_OPENMP'
    cxx_openmp_simd = true
  endif
endforeach

easysimd_native_c_flags = []
if cc.has_argument('-Wpsabi')
  easysimd_native_c_flags += '-Wno-psabi'
endif
# if cc.has_argument('-march=native+simd')
#   easysimd_native_c_flags += '-march=native+simd'
# elif cc.has_argument('-march=native')
#   easysimd_native_c_flags += '-march=native'
# endif

if cc.get_id() == 'intel'
  easysimd_native_c_flags += '-DEASYSIMD_FAST_MATH'
endif

easysimd_native_cxx_flags = []
if cxx.has_argument('-Wpsabi')
  easysimd_native_cxx_flags += '-Wno-psabi'
endif
# if cxx.has_argument('-march=native+simd')
#   easysimd_native_cxx_flags += '-march=native+simd'
# elif cxx.has_argument('-march=native')
#   easysimd_native_cxx_flags += '-march=native'
# endif

if cxx.get_id() == 'intel'
  easysimd_native_cxx_flags += '-DEASYSIMD_FAST_MATH'
endif

if not c_openmp_simd
  easysimd_deps += dependency('openmp', required: false)
endif

easysimd_deps += cc.find_library('m', required: false)

easysimd_include_dir = include_directories('..')

subdir('common')
subdir('x86')
subdir('arm')
