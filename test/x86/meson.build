easysimd_test_x86_tests = [
  'mmx',
  'sse',
  'sse2',
  'sse3',
  'ssse3',
  'sse4.1',
  'sse4.2',
  'avx',
  'avx2',
  'fma',
  'f16c',
  'gfni',
  'clmul',
  'svml',
  'xop'
]

subdir('avx512')

easysimd_test_x86_sources = []
easysimd_test_x86_libs = []
foreach name : easysimd_test_x86_tests
  foreach lang : ['c', 'cpp']
    source_file = name + '.c'
    if lang == 'cpp'
      source_file = configure_file(input: name + '.c', output: name + '.cpp', copy: true)
    endif
    easysimd_test_x86_sources += source_file

    foreach emul : ['emul', 'native']
      extra_flags = ['-DEASYSIMD_TEST_BARE']

      x = executable(name + '-' + emul +  '-' + lang, source_file,
          c_args: easysimd_c_args + easysimd_c_defs + easysimd_native_c_flags + extra_flags,
          cpp_args: easysimd_c_args + easysimd_c_defs + easysimd_native_c_flags + extra_flags,
          include_directories: easysimd_include_dir,
          dependencies: easysimd_deps)

      test('x86/' + name + '/' + emul + '/' + lang, x,
          protocol: 'tap',
          # Emscripten tests must be run from builddir
          workdir: meson.current_build_dir())
    endforeach
  endforeach
endforeach
