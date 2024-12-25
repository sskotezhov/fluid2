cmake_minimum_required(VERSION 3.10)
project(fluid)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DEFINED THREADS_NUM)
	add_compile_definitions("=fopenmp" "-DTHREADS_NUM=${THREADS_NUM}")
	add_executable(main 1.cpp)
else(DEFINED THREADS_NUM)
	message(SEND_ERROR "THREADS_NUM UNDEFINED")
endif(DEFINED THREADS_NUM)
