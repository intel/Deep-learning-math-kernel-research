Deep Learning Math Kernel Research(Euler)
============================================================

Experimental DNN math kernel based on C++11 and Intel intrinsic instructions.

## System Requirements
    Intel Core-X series processor with Intel (R) AVX-512 instruction set extensions
    Intel Xeon Scalable processor (Skylake, Cascade Lake, ...)

## Prerequisites
    Linux x86_64 OS
    CMake >= 3.0
    Boost >= 1.58.0
    Gflags >= 2.0
    [ICC >= 18.0](https://software.intel.com/en-us/c-compilers)
    (ICC >= 19.0 for Intel DL Boost (VNNI) support on Intel Cascade Lake)

## Build
    ; ICC
    source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh -arch intel64 -platform linux
    mkdir -p build && cd build
    cmake .. -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_CXX_FLAGS=-xCore-AVX512 -DWITH_TEST=ON
    make -j
    cd -

    ; Configuration of ICC 19 with VNNI support
    cmake .. -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_CXX_FLAGS=-xCore-AVX512 -DWITH_TEST=ON -DWITH_VNNI=ON

## Run Tests
    cd /path/to/euler/root
    ./scripts/best_configs/vgg-n1-8180-1s.sh 2>&1 | grep tflops

## Link to Euler
    CFLAGS += /path/to/euler/include
    #include "euler.hpp"
    LDFLAGS += libel

## License
    Apache License Version 2.0. 
