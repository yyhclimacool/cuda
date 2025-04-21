cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cuda LANGUAGES C CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CXXFLAGS
  -pipe
  -Wall
  -W
  -Wextra
  -fPIC
  -fstrict-aliasing
  -Wno-invalid-offsetof
  -Wno-unused-parameter
)
string(REPLACE ";" " " CXXFLAGS "${CXXFLAGS}")
string(APPEND CMAKE_CXX_FLAGS " ")
string(APPEND CMAKE_CXX_FLAGS ${CXXFLAGS})

message(STATUS "CMAKE_C_FLAGS_RELEASE: " ${CMAKE_C_FLAGS_RELEASE})
message(STATUS "CMAKE_C_FLAGS_RELWITHDEBINFO: " ${CMAKE_C_FLAGS_RELWITHDEBINFO})
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: " ${CMAKE_C_FLAGS_DEBUG})
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: " ${CMAKE_C_FLAGS_RELEASE})
message(STATUS "CMAKE_CXX_FLAGS_RELWITHDEBINFO: " ${CMAKE_C_FLAGS_RELWITHDEBINFO})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable(matrix_sum matrix_sum.cu)
add_executable(vector_add vector_add.cu)
add_executable(print_thread_idx print_thread_idx.cu)
set_target_properties(
  matrix_sum
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(
  vector_add print_thread_idx
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(matrix_sum glog gflags pthread)
target_link_libraries(vector_add glog gflags pthread)
target_link_libraries(print_thread_idx glog gflags pthread)
install(TARGETS matrix_sum vector_add print_thread_idx
  RUNTIME DESTINATION bin)