cmake_minimum_required(VERSION 3.18)

project(cfxmine CXX)

set(THIRDPARTY_SOURCE_DIR ${PROJECT_SOURCE_DIR}/third-party)

#if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
#  set(CMAKE_CUDA_ARCHITECTURES 61 75)
#endif()

#find_package(CUDAToolkit)

set(Boost_USE_STATIC_LIBS ON)
# set(Boost_DEBUG ON)
find_package(Boost 1.65.1 REQUIRED COMPONENTS chrono regex system thread)

set(JSONCPP_WITH_TESTS OFF CACHE BOOL "Force option off" FORCE)
set(JSONCPP_WITH_POST_BUILD_UNITTEST OFF CACHE BOOL "Force option off" FORCE)
add_subdirectory(${THIRDPARTY_SOURCE_DIR}/jsoncpp)
add_subdirectory(${THIRDPARTY_SOURCE_DIR}/tart)

add_executable(cfxmine
  src/main.cc
  src/StratumClient.cc
  src/light.cc
  src/sha3.cc
  src/OctopusCPUMiner.cc
  src/OctopusVulkanMiner.cpp
  #src/OctopusCUDAMiner.cu
)

set_property(TARGET jsoncpp_lib PROPERTY CXX_STANDARD 11)
set_property(TARGET cfxmine PROPERTY CXX_STANDARD 17)
target_include_directories(
  cfxmine
  PUBLIC
  ${THIRDPARTY_SOURCE_DIR}/cxxopts/include
)
set_target_properties(jsoncpp_lib
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)
target_link_libraries(
  cfxmine
  PUBLIC
  Boost::thread
  Boost::regex
  Boost::chrono
  tart
  #CUDA::cudart_static
  jsoncpp_lib
  #jsoncpp
)
