cmake_minimum_required(VERSION 3.30)
project(ORB)

find_package(OpenCV REQUIRED)

add_executable(main src/main.cpp src/ORB.cpp)
target_include_directories(main PRIVATE include)
target_link_libraries(main PRIVATE ${OpenCV_LIBS})
target_compile_features(main PRIVATE cxx_std_20)
