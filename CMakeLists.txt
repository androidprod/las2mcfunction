cmake_minimum_required(VERSION 3.21)
project(las2mcfunction LANGUAGES C CXX)

# -----------------------------
# Options
# -----------------------------
option(ENABLE_CUDA "Enable CUDA acceleration" ON)

# -----------------------------
# C++ settings
# -----------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------
# Sources
# -----------------------------
set(SRC
    main.cpp
    miniz.c
)

set(HEADERS
    block_palette.hpp
    miniz.h
)

# Windows resource
if (WIN32)
    set(WIN_RESOURCE
        resources/app.rc
    )
endif()

# -----------------------------
# CUDA
# -----------------------------
if (ENABLE_CUDA)
    enable_language(CUDA)
    add_definitions(-DENABLE_CUDA)

    set(CUDA_SRC
        cuda_kernel.cu
    )

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_executable(las2mcfunction
        ${SRC}
        ${HEADERS}
        ${CUDA_SRC}
        ${WIN_RESOURCE}
    )

    set_target_properties(las2mcfunction PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES native
    )

    if (MSVC)
        target_compile_options(las2mcfunction PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-allow-unsupported-compiler>
        )
    endif()

else()
    add_executable(las2mcfunction
        ${SRC}
        ${HEADERS}
        ${WIN_RESOURCE}
    )
endif()

# Try to find OpenCL and enable optional OpenCL mapping
find_path(OPENCL_INCLUDE_DIR CL/cl.h)
find_library(OPENCL_LIBRARY NAMES OpenCL OpenCL32)
if(OPENCL_INCLUDE_DIR AND OPENCL_LIBRARY)
    message(STATUS "OpenCL found: ${OPENCL_LIBRARY}")
    target_include_directories(las2mcfunction PRIVATE ${OPENCL_INCLUDE_DIR})
    target_link_libraries(las2mcfunction PRIVATE ${OPENCL_LIBRARY})
    target_compile_definitions(las2mcfunction PRIVATE USE_OPENCL=1)
    # add optional source implementing OpenCL mapping
    target_sources(las2mcfunction PRIVATE opencl_map.cpp)
else()
    message(STATUS "OpenCL not found; building without OpenCL support")
endif()

# -----------------------------
# Platform specifics
# -----------------------------
if (WIN32)
    target_compile_definitions(las2mcfunction PRIVATE
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
endif()

# -----------------------------
# Warnings
# -----------------------------
if (MSVC)
    target_compile_options(las2mcfunction PRIVATE /W4)
else()
    target_compile_options(las2mcfunction PRIVATE -Wall -Wextra -Wpedantic)
endif()
