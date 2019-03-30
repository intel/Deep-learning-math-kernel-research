# Euler: convolution kernel research

## Prerequisites
    ICC > 18.0
    (or GCC > 7.2)

## Build
    ; GCC
    mkdir -p build && cd build && cmake .. && make -j && cd -
    ; ICC
    cmake .. -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc

## Run Tests
    ; using run script
    ./script/run.sh

## Link to Euler
    CFLAGS += /path/to/euler/include
    #include "euler.hpp"
    LDFLAGS += libel

## License
    Apache License Version 2.0. 
