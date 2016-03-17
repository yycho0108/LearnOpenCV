cmake_minimum_required(VERSION 2.8)
project(Lerp)
find_package(OpenCV REQUIRED)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" C++11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" C++0X)

if(C++11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(C++0X)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
	message(STATUS "NO C++11 SUPPORT")
endif()

add_executable(Lerp Lerp.cpp)
target_link_libraries(Lerp ${OpenCV_LIBS})


