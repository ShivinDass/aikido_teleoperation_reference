find_package(catkin REQUIRED COMPONENTS
                    rospy
            )

catkin_package()

include_directories(
    include/
    ${catkin_INCLUDE_DIRS}
)

catkin_python_setup()