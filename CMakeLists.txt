cmake_minimum_required (VERSION 2.8.3)
PROJECT(patternRecognition)

option(BUILD_DOCS "Build Documentation" OFF)

find_package( OpenCV REQUIRED )

#Get c++11
ADD_DEFINITIONS(-std=c++11)

find_package(Boost COMPONENTS filesystem) 

find_package(Threads)

#Enable doxygen documentation generation
if(BUILD_DOCS)
find_package(Doxygen)
if(DOXYGEN_FOUND)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doc
${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
COMMENT "Generating API documentation with Doxygen" VERBATIM
)
endif(DOXYGEN_FOUND)

endif(BUILD_DOCS)

#Tell compiler where to find required libraries
link_directories(/usr/lib/x86_64-linux-gnu/ lib/)

include_directories(./src/library/ ./ ./jsoncpp/json ${Boost_INCLUDE_DIRS})

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

target_link_libraries(patternRecognition dl ${CMAKE_THREAD_LIBS_INIT} ${OpenCV_LIBS})

#link libraries to executable
target_link_libraries(classifierEvaluator patternRecognition boost_filesystem boost_system)

target_link_libraries(bitmapCreator patternRecognition boost_filesystem boost_system)



