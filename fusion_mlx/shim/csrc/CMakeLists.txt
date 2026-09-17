# fusion_mlx/shim/csrc/CMakeLists.txt
# C++ Shim extension build. Pattern copied from
# custom_kernels/glm_moe_dsa/csrc (nanobind + MLX CMake) with the naming
# bug fixed: metallib stem matches the C++ d.get_library() name in ALL
# three places (CMake emit, build.sh check, C++ call).
#
# PR-A ships NO Metal kernels — only the C++ Tier-1 safety base
# (hardware_probe / memory_sentinel / c_abi_envelope). The Metal block is
# present but guarded on MLX_BUILD_METAL and an empty source list so it
# is a no-op until PR-G adds the first .metal kernel.
cmake_minimum_required(VERSION 3.27)

project(fusion_mlx_shim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)

find_package(
  Python 3.8
  COMPONENTS Interpreter Development.Module
  REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m mlx --cmake-dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE MLX_PACKAGE_DIR)
# MLX ships its CMake config under <pkg>/share/cmake/MLX, not at the
# package root that `--cmake-dir` prints. Point MLX_DIR at it directly so
# find_package resolves regardless of search-path policy.
set(MLX_DIR "${MLX_PACKAGE_DIR}/share/cmake/MLX" CACHE PATH "MLX CMake config dir")
find_package(MLX CONFIG REQUIRED)
list(GET MLX_INCLUDE_DIRS 0 FUSION_SHIM_MLX_INCLUDE_DIR)

# Ops library: holds the C++ safety-base translation units. Metal kernel
# Primitive subclasses will be added here in PR-G onward.
add_library(fusion_mlx_shim_ops)
target_sources(
  fusion_mlx_shim_ops
  PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/hardware_probe.cpp
  ${CMAKE_CURRENT_LIST_DIR}/c_abi_envelope.cpp
  ${CMAKE_CURRENT_LIST_DIR}/memory_sentinel.cpp)
target_include_directories(
  fusion_mlx_shim_ops
  PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_compile_definitions(
  fusion_mlx_shim_ops
  PUBLIC FUSION_SHIM_KERNEL_SOURCE_DIR="${CMAKE_CURRENT_LIST_DIR}"
         FUSION_SHIM_MLX_INCLUDE_DIR="${FUSION_SHIM_MLX_INCLUDE_DIR}")
target_link_libraries(fusion_mlx_shim_ops PUBLIC mlx)

# Link the Metal framework on Apple for MTL::Device family queries in
# hardware_probe.cpp. Metal-cpp headers come via the MLX include path.
if(APPLE)
  find_library(METAL_FRAMEWORK Metal REQUIRED)
  target_link_libraries(fusion_mlx_shim_ops PUBLIC ${METAL_FRAMEWORK})
endif()

# Metal source compilation. PR-A has no .metal files; the block is a
# no-op but kept so PR-G can drop a .metal here and rebuild without
# touching CMake.
if(MLX_BUILD_METAL)
  set(FUSION_SHIM_METAL_SOURCES
      # PR-G will add: ${CMAKE_CURRENT_LIST_DIR}/fused_rmsnorm.metal
      #                ${CMAKE_CURRENT_LIST_DIR}/fused_rope.metal
      )
  file(GLOB_RECURSE FUSION_SHIM_METAL_HEADERS CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_LIST_DIR}/*.h)
  set(FUSION_SHIM_METAL_FLAGS
      -x metal -Wall -Wextra -fno-fast-math
      -Wno-c++17-extensions -Wno-c++20-extensions)
  if(MLX_METAL_DEBUG)
    set(FUSION_SHIM_METAL_FLAGS ${FUSION_SHIM_METAL_FLAGS}
        -gline-tables-only -frecord-sources)
  endif()
  if(NOT CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
    set(FUSION_SHIM_METAL_FLAGS ${FUSION_SHIM_METAL_FLAGS}
        "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
  endif()
  set(FUSION_SHIM_METAL_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR} ${MLX_INCLUDE_DIRS})
  set(FUSION_SHIM_METAL_INCLUDE_ARGS)
  foreach(INCLUDE_DIR IN LISTS FUSION_SHIM_METAL_INCLUDE_DIRS)
    list(APPEND FUSION_SHIM_METAL_INCLUDE_ARGS -I${INCLUDE_DIR})
  endforeach()

  set(FUSION_SHIM_METAL_AIR)
  foreach(SRCFILE IN LISTS FUSION_SHIM_METAL_SOURCES)
    get_filename_component(TARGET_NAME ${SRCFILE} NAME_WE)
    set(AIR_FILE ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.air)
    add_custom_command(
      OUTPUT ${AIR_FILE}
      COMMAND xcrun -sdk macosx metal ${FUSION_SHIM_METAL_FLAGS} -c ${SRCFILE}
              ${FUSION_SHIM_METAL_INCLUDE_ARGS} -o ${AIR_FILE}
      DEPENDS ${SRCFILE} ${FUSION_SHIM_METAL_HEADERS}
      COMMENT "Building ${TARGET_NAME}.air"
      VERBATIM)
    list(APPEND FUSION_SHIM_METAL_AIR ${AIR_FILE})
  endforeach()

  if(FUSION_SHIM_METAL_AIR)
    set(FUSION_SHIM_METALLIB
        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/fusion_mlx_shim_kernels.metallib)
    add_custom_command(
      OUTPUT ${FUSION_SHIM_METALLIB}
      COMMAND xcrun -sdk macosx metal ${FUSION_SHIM_METAL_AIR} -o
              ${FUSION_SHIM_METALLIB}
      DEPENDS ${FUSION_SHIM_METAL_AIR}
      COMMENT "Building fusion_mlx_shim_kernels.metallib"
      VERBATIM)
    add_custom_target(fusion_mlx_shim_kernels_metallib DEPENDS ${FUSION_SHIM_METALLIB})
    add_dependencies(fusion_mlx_shim_ops fusion_mlx_shim_kernels_metallib)
  endif()
endif()

# nanobind Python module. NB_DOMAIN mlx ties GC visitor registration to
# MLX's domain so mx.array args are traversed correctly.
nanobind_add_module(
  _ext
  NB_STATIC STABLE_ABI LTO NOMINSIZE NB_DOMAIN mlx
  ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp)
target_link_libraries(_ext PRIVATE fusion_mlx_shim_ops)

if(BUILD_SHARED_LIBS)
  # _ext.so lives in fusion_mlx/shim/ but links libmlx.dylib from the
  # site-packages/mlx/lib/ dir. Add rpaths so the loader finds it without
  # relying on mlx.core being imported first.
  target_link_options(
    _ext
    PRIVATE
    -Wl,-rpath,@loader_path
    -Wl,-rpath,${MLX_PACKAGE_DIR}/lib)
endif()
