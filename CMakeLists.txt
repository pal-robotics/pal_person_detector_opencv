cmake_minimum_required(VERSION 2.8.3)
project(pal_person_detector_opencv)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS image_transport
                                        cv_bridge
                                        roscpp)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES pal_person_detector_opencv
  CATKIN_DEPENDS image_transport cv_bridge roscpp
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Declare a cpp executable
add_executable(pal_person_detector_opencv src/person_detector.cpp)
target_link_libraries(pal_person_detector_opencv ${catkin_LIBRARIES} /opt/ros/hydro/lib/libopencv_ocl.so )


install(DIRECTORY include/
   DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
   FILES_MATCHING PATTERN "*.h"
)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_pal_person_detector_opencv.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
