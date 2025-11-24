cmake_minimum_required(VERSION 3.22)
project(fasttransforms CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Target name: fasttransforms.dll
add_library(fasttransforms SHARED
    src/rw_fasttransforms.cu
)

set_target_properties(fasttransforms PROPERTIES
    CUDA_ARCHITECTURES "75;80;86" # adjust for your GPU (A4500 = 86)
    POSITION_INDEPENDENT_CODE ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    OUTPUT_NAME "fasttransforms"
)

target_include_directories(fasttransforms PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(fasttransforms PRIVATE
    cufft
    cudart
)
