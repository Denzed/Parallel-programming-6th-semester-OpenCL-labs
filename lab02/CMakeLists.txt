cmake_minimum_required(VERSION 3.4)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wshadow -Wfatal-errors")

set(CMAKE_CXX_STANDARD 17)

add_executable(main main.cpp)

target_include_directories(main
	PUBLIC .
)

target_link_libraries(main
		PRIVATE OpenCL
)
