cmake_minimum_required(VERSION 2.8)
include(PICMake)
INCLUDE_DIRECTORIES(/opt/opencv-2.4.9/include)


pi_add_target(GMSTracker SHARED src REQUIRED GSLAM OpenCV PIL GLEW GLUT OpenGL)
pi_report_targets()
