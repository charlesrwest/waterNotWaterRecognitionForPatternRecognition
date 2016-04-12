cmake_minimum_required (VERSION 2.8.3)
PROJECT(patternRecognition)

option(BUILD_DOCS "Build Documentation" OFF)

find_package( OpenCV REQUIRED )

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

#Get c++11
ADD_DEFINITIONS(-std=c++11)

find_package(Boost COMPONENTS system filesystem) 

find_package(Threads)

#Tell compiler where to find required libraries
link_directories(/usr/lib/x86_64-linux-gnu/ lib/)

include_directories(./src/library/ ./ ./jsoncpp/json ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIR} ./tiny-cnn/tiny_cnn ./tiny-cnn)

#Get lists of the source files
FILE(GLOB SOURCEFILES ./src/library/*.cpp ./src/library/*.c ./jsoncpp/*.cpp)

FILE(GLOB BITMAP_CREATOR_SOURCE_FILES ./src/executables/bitmapCreator/*.cpp ./src/executables/bitmapCreator/*.c)

FILE(GLOB CLASSIFIER_EVALUATOR_SOURCE_FILES ./src/executables/classifierEvaluator/*.cpp ./src/executables/classifierEvaluator/*.c)

#Set the binaries to be placed in the ./bin/ directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin/")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib/")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib/")

#Create a libraries
add_library(patternRecognition SHARED  ${SOURCEFILES})

#Tell CMake what executables to make

ADD_EXECUTABLE(classifierEvaluator ${CLASSIFIER_EVALUATOR_SOURCE_FILES} ${CMAKE_CURRENT_BINARY_DIR}) 

ADD_EXECUTABLE(bitmapCreator ${BITMAP_CREATOR_SOURCE_FILES} ${CMAKE_CURRENT_BINARY_DIR}) 

target_link_libraries(patternRecognition dl ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS} ${Caffe_LIBRARIES})

#link libraries to executable
target_link_libraries(classifierEvaluator patternRecognition ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY})

target_link_libraries(bitmapCreator patternRecognition  ${Boost_FILESYSTEM_LIBRARY} ${Boost_SYSTEM_LIBRARY})



