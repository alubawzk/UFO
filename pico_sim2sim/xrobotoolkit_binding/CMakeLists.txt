cmake_minimum_required(VERSION 3.14)
project(ufo_xrobotoolkit_binding LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED PXREA_SDK_ROOT)
  message(FATAL_ERROR "PXREA_SDK_ROOT must point to RoboticsService/PXREARobotSDK")
endif()

set(PYBIND11_FINDPYTHON ON)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(xrobotoolkit_sdk MODULE bindings/py_bindings.cpp)
target_include_directories(xrobotoolkit_sdk PRIVATE
  "${PXREA_SDK_ROOT}"
  "${PXREA_SDK_ROOT}/nlohmann"
)
target_link_directories(xrobotoolkit_sdk PRIVATE "${PXREA_SDK_ROOT}/build")
target_link_libraries(xrobotoolkit_sdk PRIVATE PXREARobotSDK)
set_target_properties(xrobotoolkit_sdk PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)
