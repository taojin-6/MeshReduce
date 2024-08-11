add_executable(kinect_streamer
        kinect_streamer.cpp
)

target_link_libraries(kinect_streamer
        PUBLIC
        sensor
        streaming
        jsoncpp
)

add_executable(sensor_client
        sensor_client.cpp
)

target_link_libraries(sensor_client
        PUBLIC
        sensor
        reconstruction
        streaming
        jsoncpp
)
