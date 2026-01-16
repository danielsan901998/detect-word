cmake_minimum_required(VERSION 3.10)
project(detect-word CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find FFmpeg
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED 
    libavcodec 
    libavformat 
    libavutil 
    libswresample
)

# Find libraries
find_library(WHISPER_LIB whisper REQUIRED)
find_library(GGML_LIB ggml REQUIRED)
find_package(Threads REQUIRED)

add_executable(detect-word
    detect-word.cpp
    src/common.cpp
    src/common-whisper.cpp
    src/ffmpeg-transcode.cpp
)

target_include_directories(detect-word PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${FFMPEG_INCLUDE_DIRS}
)

target_compile_options(detect-word PRIVATE -O3 -march=native)
target_compile_definitions(detect-word PRIVATE WHISPER_FFMPEG)

target_link_libraries(detect-word PRIVATE
    ${WHISPER_LIB}
    ${GGML_LIB}
    ${FFMPEG_LIBRARIES}
    Threads::Threads
    dl
    m
)
