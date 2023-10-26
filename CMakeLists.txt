cmake_minimum_required(VERSION 3.18)

set(CMAKE_BUILD_TYPE "Debug")

project(RapidGKC LANGUAGES CXX CUDA)
# project(RapidGKC LANGUAGES CXX) # for CUDA optional projects
# if(CUDA_ENABLED)
#     enable_language(CUDA)
# endif()
# TO SPECIFY CUDA COMPILER, RUN WITH:
# cmake -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.1/bin/nvcc.exe"

# ** Link Pthread and set C++ std **
# set(CMAKE_CXX_FLAGS "${CAMKE_CXX_FLAGS} -std=c++14 -lpthread -lz -O2 -lstdc++fs") # NVCC will use the same flag in CMAKE_CXX_FLAGS? Seems not
set(CUDA_NVCC_FLAGS "-std=c++14 -lpthread -lz -O2 -arch=native")
# set(CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
# set(CMAKE_CUDA_STANDARD 14)
# set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

# ** Check Pthread **
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)

# ** Get CUDA Version **
find_package(CUDA REQUIRED) # required before cmake 3.9.
# ** Check CUDA Version **
set(CUDA_VERSION_MINIMUM "11.8")
if(CUDA_VERSION VERSION_LESS CUDA_VERSION_MINIMUM)
  message(FATAL_ERROR "Minimum CUDA version requirement: ${CUDA_VERSION_MINIMUM}, but found current version: ${CUDA_VERSION}.")
endif()
# ** Set CUDA Arch **
set(CMAKE_CUDA_ARCHITECTURES "native")

# ** Add Include Path **
include_directories(${PROJECT_SOURCE_DIR})

# ** Set Source Files **
file (GLOB SRC_FILES
# "${PROJECT_SOURCE_DIR}/*.h"
# "${PROJECT_SOURCE_DIR}/*.hpp"
"${PROJECT_SOURCE_DIR}/gpu_skmgen.cu"
"${PROJECT_SOURCE_DIR}/gpu_kmercounting.cu"
"${PROJECT_SOURCE_DIR}/cpu_funcs.cpp"
"${PROJECT_SOURCE_DIR}/gkc_main.cpp"
)

# set(DEFAULT_EXECUTABLE_NAME "${PROJECT_NAME}")
set(DEFAULT_EXECUTABLE_NAME "rapidgkc")
set(LONG_EXECUTABLE_NAME "l_${DEFAULT_EXECUTABLE_NAME}")
set(SHORT_EXECUTABLE_NAME "s_${DEFAULT_EXECUTABLE_NAME}")

message("${DEFAULT_EXECUTABLE_NAME}")
message("${CMAKE_THREAD_LIBS_INIT}")
message("${CMAKE_CUDA_ARCHITECTURES}")

cuda_add_executable(${DEFAULT_EXECUTABLE_NAME} ${SRC_FILES})
target_link_libraries(${DEFAULT_EXECUTABLE_NAME} z)

# cuda_add_executable(${LONG_EXECUTABLE_NAME} ${SRC_FILES})
# target_link_libraries(${LONG_EXECUTABLE_NAME} z)
# target_compile_definitions(${LONG_EXECUTABLE_NAME} PRIVATE LONGERKMER)

# cuda_add_executable(${SHORT_EXECUTABLE_NAME} ${SRC_FILES})
# target_link_libraries(${SHORT_EXECUTABLE_NAME} z)
# target_compile_definitions(${SHORT_EXECUTABLE_NAME} PRIVATE SHORTERKMER)

# add_executable()

# ** Post Build Process Example **
# add_custom_command(
#     TARGET ${CMAKE_PROJECT_NAME}
#     POST_BUILD
#     COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_PROJECT_NAME}
# )