# Install OpenCV http://milq.github.io/install-opencv-ubuntu-debian/
cmake_minimum_required(VERSION 3.16)
project(DPDA_Executable)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(DPDA_Executable main.cpp)
target_link_libraries(DPDA_Executable ${OpenCV_LIBS})

find_package(Boost 1.71.0 COMPONENTS filesystem regex)

if(Boost_FOUND)
    target_link_libraries(DPDA_Executable Boost::filesystem Boost::regex)
    include_directories( ${Boost_INCLUDE_DIRS} )
endif()

