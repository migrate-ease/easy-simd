


function build_with_asan()
{
    echo "build with asan flag"
    CFLAGS="-march=armv8.5-a+crc+sve2+sha2+sha3+sve2-sha3+sve2-aes+sve2-bitperm  -msve-vector-bits=128  -fsanitize=address -fsanitize=undefined" \
    CXXFLAGS="-march=armv8.5-a+crc+sve2+sha2+sha3+sve2-sha3+sve2-aes+sve2-bitperm  -msve-vector-bits=128  -fsanitize=address -fsanitize=undefined" \
    LDFLAGS="-fsanitize=address -fsanitize=undefined" \
    meson build
    ninja -C build
    ninja -C build test
}

function build_all()
{
    echo "build all"
    CFLAGS="-march=armv8.5-a+crc+sve2+sha2+sha3+sve2-sha3+sve2-aes+sve2-bitperm -msve-vector-bits=128" \
    CXXFLAGS="-march=armv8.5-a+crc+sve2+sha2+sha3+sve2-sha3+sve2-aes+sve2-bitperm -msve-vector-bits=128" \
    meson build
    ninja -C build
}

function build_test()
{
    echo "build and test"
    CFLAGS="-march=armv8.5-a+crc+sve2+sha2+sha3+sve2-sha3+sve2-aes+sve2-bitperm -msve-vector-bits=128 " \
    CXXFLAGS="-march=armv8.5-a+crc+sve2+sha2+sha3+sve2-sha3+sve2-aes+sve2-bitperm -msve-vector-bits=128 " \
    meson build
    ninja -C build test
}

function build_clean
{
    rm -rf build
    rm -rf build-clang
}

function build_benchmark
{
    echo "build benchmark"
    CFLAGS="-O0 -march=armv8.5-a+crc+sve2+sha2+sha3+sve2-sha3+sve2-aes+sve2-bitperm -DEASYSIMD_ENABLE_TEST_PERF -DCLOCK_THREAD_CPUTIME_ID -msve-vector-bits=128" \
    CXXFLAGS="-O0 -march=armv8.5-a+crc+sve2+sha2+sha3+sve2-sha3+sve2-aes+sve2-bitperm -DEASYSIMD_ENABLE_TEST_PERF -DCLOCK_THREAD_CPUTIME_ID  -msve-vector-bits=128" \
    meson build
    ninja -C build
}

function build_on_x86_64
{
    echo "build on _x86_64"
    CFLAGS="-march=native -msse -msse2 -msse3 -msse3 -msse4.1 -msse4.2 -mavx -mavx2 -mavx512f -mavx512vl -mavx512dq -mavx512bw" \
    meson build
    ninja -C build
}

function build_neon()
{
    echo "build and test with only neon"
    CFLAGS="-march=armv8.2-a+crc+crypto" \
    CXXFLAGS="-march=armv8.2-a+crc+crypto" \
    meson build
    ninja -C build test
}

function build_clang()
{
    echo "build and test with clang"
    CC=clang \
    CXX=clang++ \
    CFLAGS="-g -O2 -march=armv8.2-a+crc+crypto" \
    CXXFLAGS="-g -O2 -march=armv8.2-a+crc+crypto" \
    meson build-clang
    ninja -C build-clang
    ninja -C build-clang test
}

function usage()
{
    printf "Usage:\n"
    printf "\t./build.sh all \n"
    printf "\t./build.sh test \n"
    printf "\t./build.sh asan \n"
    printf "\t./build.sh benchmark \n"
    printf "\t./build.sh on_x86_64 \n"
    printf "\t./build.sh only_neon \n"
    printf "\t./build.sh clang \n"
    printf "\t./build.sh clean \n"
}

while [ -n "$1" ]; do
    case "$1" in
	asan )
	    build_with_asan
	    shift ;;
	all )
	    build_all
	    shift ;;
	test )
	    build_test
	    shift ;;
	benchmark )
	    build_benchmark
	    shift ;;
	on_x86_64 )
	    build_on_x86_64
	    shift ;;
	only_neon )
	    build_neon
	    shift ;;
	clang )
	    build_clang
	    shift ;;
	clean )
	    build_clean
	    shift ;;
	* )
	    usage
	    shift ;;
    esac
done