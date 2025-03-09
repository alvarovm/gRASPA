cmake_minimum_required(VERSION 3.10)

# Set the project name
project(gRASPA CXX)



set(CMAKE_VERBOSE_MAKEFILE ON) #
option(USE_SYCL "Using SYCL backend" ON)
if (USE_SYCL)
set(CMAKE_CXX_COMPILER "icpx")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -g -O3 -finline -qopenmp")
#else()
#  project (gpu4pyscf C CXX CUDA Fortran)
endif()

if (USE_SYCL)
find_package(IntelSYCL)
endif()

if(USE_SYCL)
add_definitions(-DUSE_SYCL=1)
endif(USE_SYCL)

# Set the C++ standard
#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Uncomment the following line if you want to include FPGA_FLAG
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fintelfpga")

#include_directories(${PROJECT_SOURCE_DIR}/src)

add_subdirectory(src)

