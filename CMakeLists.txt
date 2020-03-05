 cmake_minimum_required(VERSION 3.0)
 project(example)
 
 find_package(Torch REQUIRED)
 find_package(OpenCV 4.1.0 REQUIRED)
 
 include_directories(${OpenCV_INCLUDE_DIRS})
 include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
 
 link_directories(${CMAKE_BINARY_DIR}/bin)
 set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
 
 add_executable(example src/main.cpp src/dataset.cpp)
 
 target_link_libraries(example ${OpenCV_LIBS})
 target_link_libraries(example "${TORCH_LIBRARIES}")
 
 set_property(TARGET example PROPERTY CXX_STANDARD 11)

