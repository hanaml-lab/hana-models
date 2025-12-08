cmake_minimum_required(VERSION 3.22)
project(llama_bridge CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
# llama.cpp, pinned. On Android, configure with -DGGML_VULKAN=ON (see android/app/build.gradle.kts).
FetchContent_Declare(
  llama
  GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
  GIT_TAG        b6595
  GIT_SHALLOW    TRUE
)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(llama)

add_library(llama_bridge SHARED llama_bridge.cpp)

if(ANDROID)
  target_link_libraries(llama_bridge PRIVATE llama android log)
else()
  # Host builds, for checking the bridge compiles: JNI headers from the JDK.
  find_package(JNI REQUIRED)
  target_include_directories(llama_bridge PRIVATE ${JNI_INCLUDE_DIRS})
  target_link_libraries(llama_bridge PRIVATE llama)
endif()
