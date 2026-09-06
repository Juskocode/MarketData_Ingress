cmake_minimum_required(VERSION 3.18)

project(MarketData_Ingress LANGUAGES CXX)

option(MD_WITH_TENSORRT "Build with TensorRT when available" ON)
option(MD_BUILD_TESTS "Build automated tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(core_objects OBJECT
  src/benchmark.cpp
  src/engine_factory.cpp
)
target_include_directories(core_objects PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(HAVE_TENSORRT OFF)
if(MD_WITH_TENSORRT)
  find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS
      /usr/include
      /usr/include/x86_64-linux-gnu
      /usr/local/include
      $ENV{CUDA_HOME}/include
      /usr/local/cuda/include
  )
  find_library(TENSORRT_LIBRARY NAMES nvinfer
    PATHS
      /usr/lib/x86_64-linux-gnu
      /usr/local/lib
      /usr/local/cuda/lib64
      /usr/lib
      /usr/local/cuda/lib
  )
  find_library(TENSORRT_PARSER_LIBRARY NAMES nvonnxparser
    PATHS
      /usr/lib/x86_64-linux-gnu
      /usr/local/lib
      /usr/local/cuda/lib64
      /usr/lib
      /usr/local/cuda/lib
  )
  find_library(CUDA_RUNTIME_LIBRARY NAMES cudart
    PATHS
      /usr/lib/x86_64-linux-gnu
      /usr/local/cuda/lib64
      /usr/local/lib
  )

  if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY AND TENSORRT_PARSER_LIBRARY AND CUDA_RUNTIME_LIBRARY)
    set(HAVE_TENSORRT ON)
    message(STATUS "TensorRT detected; building native TensorRT backend")
    add_library(inference_backend src/engine_tensorrt.cpp)
    target_include_directories(inference_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(inference_backend
      PRIVATE
        ${TENSORRT_LIBRARY}
        ${TENSORRT_PARSER_LIBRARY}
        ${CUDA_RUNTIME_LIBRARY}
    )
    target_compile_definitions(inference_backend PRIVATE MD_WITH_TENSORRT)
    target_compile_definitions(core_objects PRIVATE MD_WITH_TENSORRT)
  else()
    message(WARNING "TensorRT requested but not detected; using mock backend")
  endif()
endif()

if(NOT HAVE_TENSORRT)
  add_library(inference_backend src/engine_mock.cpp)
  target_include_directories(inference_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

target_link_libraries(core_objects PUBLIC inference_backend)

add_executable(marketdata_ingress src/main.cpp)
target_link_libraries(marketdata_ingress PRIVATE core_objects inference_backend)

if(MD_BUILD_TESTS)
  enable_testing()
  add_executable(marketdata_tests
    tests/test_smoke.cpp
    src/benchmark.cpp
    src/engine_factory.cpp
  )
  target_include_directories(marketdata_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_definitions(marketdata_tests PRIVATE MD_FORCE_MOCK_BACKEND)
  if(NOT HAVE_TENSORRT)
    target_link_libraries(marketdata_tests PRIVATE
      inference_backend
    )
  else()
    target_sources(marketdata_tests PRIVATE src/engine_mock.cpp)
  endif()
  add_test(NAME smoke_benchmark COMMAND marketdata_tests)
endif()

install(TARGETS marketdata_ingress RUNTIME DESTINATION bin)
