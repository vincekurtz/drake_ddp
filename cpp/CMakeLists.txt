cmake_minimum_required(VERSION 3.10)
project(drake_ddp)

# Use stricter warnings
if(MSVC)
  add_compile_options(/W4 /WX)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Include drake as an external library
# Note that $CMAKE_PREFIX_PATH must be set accordingly
find_package(drake CONFIG REQUIRED)

# Add executables
add_executable(simple_ball src/simple_ball.cpp)

# Link external libraries
target_link_libraries(simple_ball drake::drake)
