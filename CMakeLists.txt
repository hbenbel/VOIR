cmake_minimum_required(VERSION 3.0)

project (Tracker)

file(GLOB_RECURSE Tracker_SOURCES "src/*.cpp")
file(GLOB_RECURSE Tracker_HEADERS "src/*.hpp")

find_package(OpenCV REQUIRED)

set (Tracker_INCLUDE_DIRS "")
foreach (_headerFile ${Tracker_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list (APPEND Tracker_INCLUDE_DIRS ${_dir})
endforeach()
list(REMOVE_DUPLICATES Tracker_INCLUDE_DIRS)

add_executable (Tracker ${Tracker_SOURCES})
target_include_directories(Tracker PRIVATE ${Tracker_INCLUDE_DIRS})
target_link_libraries(Tracker ${OpenCV_LIBS})