#include <cxxabi.h>
#include <chrono>

#ifndef __ELT_UTILS_HPP__
#define __ELT_UTILS_HPP__

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float, std::milli> Duration;

#define time_start(name) \
    printf("Timing:%s: ", #name); \
    Time::time_point __s##name =  Time::now();
#define time_end(name, iterations) \
    Time::time_point __e##name = Time::now(); \
    printf("%.2f ms\n", Duration(__e##name - __s##name).count() / iterations);

namespace euler {


}

#endif // __ELT_UTILS_HPP__
