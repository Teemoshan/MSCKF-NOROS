cmake_minimum_required(VERSION 2.8)
project(MSCKF_NOROS)

set(CMAKE_BUILD_TYPE "Release")
add_compile_options(-std=c++11)


find_package(Boost REQUIRED COMPONENTS math_c99 random)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 3 REQUIRED)


include_directories(
  include
  ${PROJECT_SOURCE_DIR}/include/msckf_vio
  ${EIGEN3_INCLUDE_DIR}
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Msckf Vio
add_library(${PROJECT_NAME} SHARED
    src/msckf_vio.cpp
    src/utils.cpp
    src/image_processor.cpp
    src/random_numbers.cpp
)

target_link_libraries(${PROJECT_NAME} 
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
)

add_executable(msckf_noros   src/main.cpp)

target_link_libraries(msckf_noros  ${PROJECT_NAME} /usr/lib/x86_64-linux-gnu/libpthread.so )


