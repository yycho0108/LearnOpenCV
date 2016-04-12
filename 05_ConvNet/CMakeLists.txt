cmake_minimum_required(VERSION 2.8)
project(ConvNet)
find_package(OpenCV REQUIRED)
find_package(Threads)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
	message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(SOURCES
	${SOURCES}
	${CMAKE_CURRENT_SOURCE_DIR}/Utility.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Parser.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ActivationLayer.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/DenseLayer.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/FlattenLayer.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/SoftMaxLayer.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/PoolLayer.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ConvolutionLayer.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/ConvNet.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

set(HEADERS
	${HEADERS}
	${CMAKE_CURRENT_SOURCE_DIR}/Utility.h
	${CMAKE_CURRENT_SOURCE_DIR}/Parser.h
	${CMAKE_CURRENT_SOURCE_DIR}/Layer.h
	${CMAKE_CURRENT_SOURCE_DIR}/ActivationLayer.h
	${CMAKE_CURRENT_SOURCE_DIR}/DenseLayer.h
	${CMAKE_CURRENT_SOURCE_DIR}/FlattenLayer.h
	${CMAKE_CURRENT_SOURCE_DIR}/SoftMaxLayer.h
	${CMAKE_CURRENT_SOURCE_DIR}/PoolLayer.h
	${CMAKE_CURRENT_SOURCE_DIR}/ConvolutionLayer.h
	${CMAKE_CURRENT_SOURCE_DIR}/ConvNet.h
)

add_executable(ConvNet ${SOURCES} ${HEADERS})

target_link_libraries(ConvNet ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})
