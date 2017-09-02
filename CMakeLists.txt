cmake_minimum_required(VERSION 2.8)
include(PICMake)
pi_add_target(GMSTracker SHARED GMSTracker.cpp REQUIRED GSLAM OpenCV)
pi_report_targets()
