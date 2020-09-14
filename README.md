Deep Learning Math Kernel Research (Euler)
============================================================

Experimental DNN math kernel based on C++11 and Intel intrinsic instructions.

## System Requirements
    Intel Core-X series processor with Intel (R) AVX-512 instruction set extensions
    Intel Xeon Scalable processor (Skylake, Cascade Lake, ...)

## Prerequisites
    Linux x86_64 OS
    CMake >= 3.0
    Gflags >= 2.0
    [ICC >= 18.0](https://software.intel.com/en-us/c-compilers)
    (ICC >= 19.0 for Intel DL Boost (VNNI) support on Intel Cascade Lake)

## Build
    ; ICC/ICX
    source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh -arch intel64 -platform linux
    mkdir -p build && cd build
    ; ICC
    cmake .. -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_CXX_FLAGS=-xCore-AVX512 -DWITH_TEST=ON
    ; ICX
    cmake .. -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DCMAKE_CXX_FLAGS=-xCore-AVX512 -DWITH_TEST=ON
    make -j
    cd -

    ; CMake build option to enable VNNI support (default: OFF)
    -DWITH_VNNI=ON
    ; CMake build option to enable Intel TBB threading runtime (default: OMP)
    -DMT_RUNTIME=TBB
    ; CMake build option to enable FP16 user inputs (default: OFF)
    -DENABLE_USER_FP16=ON

## Run Tests
    cd /path/to/euler/root
    ./scripts/best_configs/vgg-n1-8180-1s.sh

## Link to Euler
    CFLAGS += /path/to/euler/include
    #include "euler.hpp"
    LDFLAGS += libel

## License
    Apache License Version 2.0. 
