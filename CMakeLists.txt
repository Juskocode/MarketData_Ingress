cmake_minimum_required(VERSION 3.20)

project(MarketData_Ingress VERSION 0.2.0 LANGUAGES CXX)

set(MD_BUILD_REVISION "$ENV{GITHUB_SHA}" CACHE STRING "Build revision embedded in metrics")
if(MD_BUILD_REVISION STREQUAL "")
  set(MD_BUILD_REVISION "unknown" CACHE STRING "Build revision embedded in metrics" FORCE)
endif()
if(NOT MD_BUILD_REVISION MATCHES "^[A-Za-z0-9._-]+$")
  message(FATAL_ERROR "MD_BUILD_REVISION may contain only letters, digits, dot, underscore, and dash")
endif()

option(MD_WITH_TENSORRT "Build the TensorRT backend when TensorRT 10+ is available" ON)
option(MD_REQUIRE_TENSORRT "Fail configuration unless TensorRT 10+ is available" OFF)
option(MD_BUILD_TESTS "Build automated tests" ON)
option(MD_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(GNUInstallDirs)

add_library(md_project_options INTERFACE)
target_compile_features(md_project_options INTERFACE cxx_std_17)
target_compile_definitions(md_project_options INTERFACE
  MD_PROJECT_VERSION="${PROJECT_VERSION}"
  MD_BUILD_REVISION="${MD_BUILD_REVISION}"
)
if(MSVC)
  target_compile_options(md_project_options INTERFACE /W4 /permissive-)
else()
  target_compile_options(md_project_options INTERFACE
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
  )
  if(MD_ENABLE_SANITIZERS)
    target_compile_options(md_project_options INTERFACE
      -fsanitize=address,undefined -fno-omit-frame-pointer
    )
    target_link_options(md_project_options INTERFACE
      -fsanitize=address,undefined -fno-omit-frame-pointer
    )
  endif()
endif()

if(MD_REQUIRE_TENSORRT)
  set(MD_WITH_TENSORRT ON)
endif()

set(HAVE_TENSORRT OFF)
if(MD_WITH_TENSORRT)
  find_package(CUDAToolkit QUIET)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${CUDAToolkit_INCLUDE_DIRS} $ENV{TENSORRT_ROOT}/include
    PATHS /usr/include /usr/include/x86_64-linux-gnu /usr/local/include /usr/local/cuda/include
  )
  find_library(TENSORRT_LIBRARY NAMES nvinfer
    HINTS $ENV{TENSORRT_ROOT}/lib $ENV{TENSORRT_ROOT}/lib64
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/local/cuda/lib64 /usr/lib
  )
  find_library(TENSORRT_PARSER_LIBRARY NAMES nvonnxparser
    HINTS $ENV{TENSORRT_ROOT}/lib $ENV{TENSORRT_ROOT}/lib64
    PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib /usr/local/cuda/lib64 /usr/lib
  )

  set(TENSORRT_VERSION_MAJOR 0)
  if(TENSORRT_INCLUDE_DIR AND EXISTS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h")
    file(STRINGS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" TENSORRT_MAJOR_LINE
      REGEX "^#define NV_TENSORRT_MAJOR +[0-9]+"
    )
    string(REGEX MATCH "[0-9]+" TENSORRT_VERSION_MAJOR "${TENSORRT_MAJOR_LINE}")
  endif()

  if(CUDAToolkit_FOUND
     AND TENSORRT_INCLUDE_DIR
     AND TENSORRT_LIBRARY
     AND TENSORRT_PARSER_LIBRARY
     AND TENSORRT_VERSION_MAJOR GREATER_EQUAL 10)
    set(HAVE_TENSORRT ON)
    message(STATUS "TensorRT ${TENSORRT_VERSION_MAJOR}.x detected; enabling native GPU backend")
  elseif(MD_REQUIRE_TENSORRT)
    message(FATAL_ERROR
      "MD_REQUIRE_TENSORRT=ON requires CUDA Toolkit and TensorRT 10+ (nvinfer + nvonnxparser)"
    )
  else()
    message(WARNING "TensorRT 10+ was not detected; building the deterministic mock backend")
  endif()
endif()

if(HAVE_TENSORRT)
  add_library(inference_backend
    src/engine_tensorrt.cpp
    src/engine_mock.cpp
  )
  target_include_directories(inference_backend PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TENSORRT_INCLUDE_DIR}
  )
  target_compile_definitions(inference_backend PUBLIC MD_WITH_TENSORRT)
  target_link_libraries(inference_backend
    PRIVATE
      md_project_options
      ${TENSORRT_LIBRARY}
      ${TENSORRT_PARSER_LIBRARY}
      CUDA::cudart
  )
else()
  add_library(inference_backend src/engine_mock.cpp)
  target_include_directories(inference_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(inference_backend PRIVATE md_project_options)
endif()

add_library(core_objects OBJECT
  src/benchmark.cpp
  src/engine_factory.cpp
)
target_include_directories(core_objects PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(core_objects PUBLIC inference_backend md_project_options)

add_executable(marketdata_ingress src/main.cpp)
target_link_libraries(marketdata_ingress PRIVATE core_objects inference_backend md_project_options)

if(HAVE_TENSORRT)
  add_executable(marketdata_build_engine src/build_tiny_engine.cpp)
  target_include_directories(marketdata_build_engine PRIVATE ${TENSORRT_INCLUDE_DIR})
  target_link_libraries(marketdata_build_engine
    PRIVATE
      md_project_options
      ${TENSORRT_LIBRARY}
      CUDA::cudart
  )
endif()

if(MD_BUILD_TESTS)
  enable_testing()
  add_executable(marketdata_tests
    tests/test_smoke.cpp
    src/benchmark.cpp
    src/engine_factory.cpp
    src/engine_mock.cpp
  )
  target_include_directories(marketdata_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_definitions(marketdata_tests PRIVATE MD_FORCE_MOCK_BACKEND)
  target_link_libraries(marketdata_tests PRIVATE md_project_options)

  add_test(NAME unit_metrics COMMAND marketdata_tests)
  add_test(NAME cli_help COMMAND marketdata_ingress --help)
  add_test(NAME cli_mock_benchmark COMMAND marketdata_ingress
    --backend mock --model mock --iterations 16 --warmup 4 --target-us 10000
  )
endif()

install(TARGETS marketdata_ingress RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(HAVE_TENSORRT)
  install(TARGETS marketdata_build_engine RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(FILES README.md DESTINATION ${CMAKE_INSTALL_DOCDIR})
install(DIRECTORY docs/ DESTINATION ${CMAKE_INSTALL_DOCDIR})
install(DIRECTORY tools/visualization/
  DESTINATION ${CMAKE_INSTALL_DATADIR}/marketdata-ingress/dashboard
)

set(CPACK_GENERATOR TGZ)
set(CPACK_PACKAGE_NAME "marketdata-ingress")
set(CPACK_PACKAGE_VENDOR "Juskocode")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Native C++ TensorRT edge-inference benchmark")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_FILE_NAME
  "${CPACK_PACKAGE_NAME}-${PROJECT_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}"
)
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)
add_executable(marketdata_metrics_compare tools/metrics_compare.cpp)
target_compile_features(marketdata_metrics_compare PRIVATE cxx_std_17)

if(MSVC)
  target_compile_options(marketdata_metrics_compare PRIVATE /W4)
else()
  target_compile_options(marketdata_metrics_compare PRIVATE -Wall -Wextra -Wpedantic)
endif()

install(TARGETS marketdata_metrics_compare RUNTIME DESTINATION bin)

add_test(
  NAME metrics_compare_self_test
  COMMAND marketdata_metrics_compare --self-test
)

add_test(
  NAME cli_metrics_regression
  COMMAND bash
          ${CMAKE_CURRENT_SOURCE_DIR}/scripts/metrics_regression_smoke.sh
          $<TARGET_FILE:marketdata_ingress>
          $<TARGET_FILE:marketdata_metrics_compare>
          ${CMAKE_CURRENT_BINARY_DIR}/metrics-regression-smoke
)

include(CPack)
