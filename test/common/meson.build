easysimd_test_common_tests = [
  'common'
]

easysimd_test_common_sources = []
foreach name : easysimd_test_common_tests
  foreach lang : ['c', 'cpp']
    source_file = name + '.c'
    if lang == 'cpp'
      source_file = configure_file(input: name + '.c', output: name + '.cpp', copy: true)
    endif
    easysimd_test_common_sources += source_file

    extra_flags = ['-DEASYSIMD_TEST_BARE']

    x = executable(name + '-' + lang, source_file,
        c_args: easysimd_c_args + easysimd_c_defs + easysimd_native_c_flags + extra_flags,
        cpp_args: easysimd_c_args + easysimd_c_defs + easysimd_native_c_flags + extra_flags,
        include_directories: easysimd_include_dir,
        dependencies: easysimd_deps)

    test('common/' + name + '/' + lang, x,
        protocol: 'tap',
        # Emscripten tests must be run from builddir
        workdir: meson.current_build_dir())
  endforeach
endforeach
