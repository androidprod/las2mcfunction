# ============================================================
# las2mcfunction
# Linux / WSL2 / Docker 対応
# CPU / CUDA / OpenCL 自動切り替え CMakeLists
# ============================================================

cmake_minimum_required(VERSION 3.21)
project(las2mcfunction LANGUAGES C CXX)

# ------------------------------------------------------------
# Options
# ------------------------------------------------------------
option(ENABLE_CUDA   "Enable CUDA acceleration if available" ON)
option(ENABLE_OPENCL "Enable OpenCL acceleration if available" ON)

# ------------------------------------------------------------
# C / C++ settings
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
endif()

# ------------------------------------------------------------
# Sources
# ------------------------------------------------------------
set(SRC
    main.cpp
    miniz.c
)

set(HEADERS
    block_palette.hpp
    miniz.h
)

# ------------------------------------------------------------
# Executable (base: CPU only)
# ------------------------------------------------------------
add_executable(las2mcfunction
    ${SRC}
    ${HEADERS}
)

target_compile_definitions(las2mcfunction PRIVATE USE_CPU=1)

# ------------------------------------------------------------
# CUDA (Linux / WSL2 / Docker safe)
# ------------------------------------------------------------
set(USE_CUDA FALSE)

if (ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(USE_CUDA TRUE)

        message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")

        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        target_sources(las2mcfunction PRIVATE
            cuda_kernel.cu
        )

        target_compile_definitions(las2mcfunction PRIVATE
            ENABLE_CUDA=1
            USE_CUDA=1
        )

        set_target_properties(las2mcfunction PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
        )

        # Docker / WSL2 / 古いnvcc対策
        if (DEFINED CMAKE_CUDA_ARCHITECTURES)
            set_property(TARGET las2mcfunction PROPERTY
                CUDA_ARCHITECTURES native
            )
        endif()

    else()
        message(WARNING "CUDA not found -> building without CUDA")
    endif()
endif()

# ------------------------------------------------------------
# OpenCL (Linux / WSL2 / Docker)
# ------------------------------------------------------------
set(USE_OPENCL FALSE)

if (ENABLE_OPENCL)
    find_package(OpenCL QUIET)

    if (OpenCL_FOUND)
        set(USE_OPENCL TRUE)

        message(STATUS "OpenCL found")

        target_sources(las2mcfunction PRIVATE
            opencl_map.cpp
        )

        target_link_libraries(las2mcfunction PRIVATE
            OpenCL::OpenCL
        )

        target_compile_definitions(las2mcfunction PRIVATE
            ENABLE_OPENCL=1
            USE_OPENCL=1
        )
    else()
        message(STATUS "OpenCL not found -> building without OpenCL")
    endif()
endif()

# ------------------------------------------------------------
# Platform-specific (Linux / WSL2 / Docker)
# ------------------------------------------------------------
if (UNIX AND NOT APPLE)
    target_link_libraries(las2mcfunction PRIVATE
        pthread
        dl
        m
    )
endif()

# ------------------------------------------------------------
# Warnings
# ------------------------------------------------------------
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(las2mcfunction PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

# ------------------------------------------------------------
# Build summary
# ------------------------------------------------------------
message(STATUS "========== Build Configuration ==========")
message(STATUS "CPU support     : ENABLED")
message(STATUS "CUDA support    : ${USE_CUDA}")
message(STATUS "OpenCL support  : ${USE_OPENCL}")
message(STATUS "=========================================")

