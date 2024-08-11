set(CMAKE_PREFIX_PATH /home/built_packages/open3d_install_0.18.0)

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(MeshReduce VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)

find_package(jsoncpp REQUIRED)
find_package(Open3D 0.18.0 REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
find_package(OpenCV REQUIRED
        core imgproc imgcodecs highgui calib3d)

include_directories(include)

add_subdirectory(src)

add_subdirectory(app)

link_libraries(sensor reconstruction jsoncpp streaming)
