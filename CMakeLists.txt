cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_COMPILER /usr/bin/clang++)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
project(dendrite-cpp CXX)

# Adding Raylib
include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)
# May include raylib for visualization/interactivity later
# set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE) # don't build the supplied examples
# set(BUILD_GAMES    OFF CACHE BOOL "" FORCE) # don't build the supplied example games

# FetchContent_Declare(
#     raylib
#     GIT_REPOSITORY "https://github.com/raysan5/raylib.git"
#     GIT_TAG "master"
#     GIT_PROGRESS TRUE
# )
#
# FetchContent_MakeAvailable(raylib)
#
# Adding our source files
file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp") # Define PROJECT_SOURCES as a list of all source files
set(PROJECT_INCLUDE "${CMAKE_CURRENT_LIST_DIR}/src/") # Define PROJECT_INCLUDE to be the path to the include directory of the project

# Declaring our executable
add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_INCLUDE})
# target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -g)
target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic -O3 -g)

# target_link_libraries(${PROJECT_NAME} PRIVATE raylib)
