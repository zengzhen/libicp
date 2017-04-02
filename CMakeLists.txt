# project
cmake_minimum_required (VERSION 2.6)
project (libicp)

SET(CMAKE_MODULE_PATH "/usr/share/cmake-2.8/Modules/")
message(STATUS "module path: ${CMAKE_MODULE_PATH} !!!")

## Eigen3
find_package(Eigen3 REQUIRED)
if(EIGEN3_INCLUDE_DIR)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif(EIGEN3_INCLUDE_DIR)

# directories
set (LIBICP_SRC_DIR src)
FIND_PATH(BOOST_DIR "boost")

# include directory
include_directories("${LIBICP_SRC_DIR}")
include_directories("${BOOST_DIR}")

# determine if OpenMP can/should be used
option(USE_OPENMP "Enable OpenMP?" ON) # set to OFF to disable
if(USE_OPENMP)
 FIND_PACKAGE(OpenMP)
 if(OPENMP_FOUND OR OpenMP_FOUND)
  message(STATUS "OpenMP flags = ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
 else(OPENMP_FOUND OR OpenMP_FOUND)
  message(STATUS "OpenMP not supported")
 endif(OPENMP_FOUND OR OpenMP_FOUND)
endif(USE_OPENMP)

# sources
FILE(GLOB LIBICP_SRC_FILES "src/*.cpp")

add_library(${PROJECT_NAME} SHARED
  src/icp.cpp
  src/icpPointToPlane.cpp
  src/icpPointToPoint.cpp
  src/kdtree.cpp
  src/matrix.cpp
)

# make release version
set(CMAKE_BUILD_TYPE Release)

# build demo program
# add_executable(icp ${LIBICP_SRC_FILES})
add_executable(icp src/demo.cpp)
add_executable(icp_projective src/demo_projective.cpp)
target_link_libraries(icp
 ${PROJECT_NAME}
)
target_link_libraries(icp_projective
 ${PROJECT_NAME}
)

