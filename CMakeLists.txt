cmake_minimum_required(VERSION 2.8.3)
project(pal_person_detector_opencv)

find_package(catkin REQUIRED COMPONENTS image_transport
                                        pal_detection_msgs
                                        cv_bridge
                                        roscpp)
find_package(OpenCV REQUIRED)


catkin_package(
  CATKIN_DEPENDS image_transport cv_bridge pal_detection_msgs roscpp
)

include_directories(SYSTEM ${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_executable(pal_person_detector_opencv src/person_detector.cpp)
add_dependencies(pal_person_detector_opencv ${catkin_EXPORTED_TARGETS})
target_link_libraries(pal_person_detector_opencv ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

install(TARGETS pal_person_detector_opencv
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
