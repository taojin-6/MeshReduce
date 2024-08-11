#include <jsoncpp/json/json.h>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <string>
#include <thread>
#include <chrono>

#include "pipe.hpp"
#include "reconstruction/reconstruction.hpp"
#include "sensor/kinect_cameras.hpp"
#include "streaming/send_image.hpp"

static std::atomic<bool> stop_consumer = false;
int variance_threshold = 20;

static void sigint_handler(int)
{
  stop_consumer = true;
}

bool blur_detection(cv::Mat grayimage)
{
  cv::Mat laplacian;
  cv::Laplacian(grayimage, laplacian, CV_8U);
  cv::Scalar mean, stddev;
  cv::meanStdDev(laplacian, mean, stddev);
  double variance = stddev.val[0] * stddev.val[0];

  std::cout << "variance: " << variance << std::endl;

  return variance < variance_threshold;
}

void image_consumer(PipeDataInCollection<ImageFrame>* dImageFrame, 
                    SenderToPython* sender)
{
  int socket = sender->get_socket();
  while (!stop_consumer)
  {
    for (size_t idx = 0; idx < dImageFrame->get_size(); idx++)
    {
      ImageFrame frame = dImageFrame->fetch(idx);

      // Process the frame here
      cv::Mat col_image = frame.cv_colImg;
      cv::Mat dep_image = frame.cv_depImg;
      cv::Mat gray_img;
    
      cv::cvtColor(col_image, gray_img, cv::COLOR_BGR2GRAY);
      bool is_blur = blur_detection(gray_img);
      if (is_blur)
      {
        std::cout << "blur detected" << std::endl;
        continue;
      }

      // Encode color and depth images
      std::vector<uchar> gray_img_buf, dep_img_buf;
      cv::imencode(".png", gray_img, gray_img_buf);
      cv::imencode(".png", dep_image, dep_img_buf);
      size_t total_img_size = gray_img_buf.size() + dep_img_buf.size() + sizeof(size_t) * 2;

      std::vector<uchar> send_buf(total_img_size);

      // Copy sizes and data to buffer
      size_t offset = 0;
      size_t gray_image_size = gray_img_buf.size();
      size_t dep_image_size = dep_img_buf.size();
      memcpy(send_buf.data() + offset, &gray_image_size, sizeof(size_t));
      offset += sizeof(size_t);
      memcpy(send_buf.data() + offset, gray_img_buf.data(), gray_img_buf.size());
      offset += gray_img_buf.size();
      memcpy(send_buf.data() + offset, &dep_image_size, sizeof(size_t));
      offset += sizeof(size_t);
      memcpy(send_buf.data() + offset, dep_img_buf.data(), dep_img_buf.size());

      // Send send_buf size first
      send(socket, &total_img_size, sizeof(total_img_size), 0);
      // Send send_buf
      send(socket, send_buf.data(), send_buf.size(), 0);

      std::cout << "total image size: " << total_img_size / 1024.0 / 1024.0 << " MB" << std::endl;

      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }
}

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    std::cerr << "usage: " << argv[0] << " <config-json>\n";
    return 1;
  }

  Json::Value jsonConf;
  {
    std::ifstream fs(argv[1]);
    if (!(fs >> jsonConf))
      std::cerr << "Error reading config\n";
  }
  int device_count = jsonConf["kinect_config"]["device_count"].asInt();
  std::cout << jsonConf["kinect_config"]["master_serial"].asString()
            << std::endl;

  PipeDataInCollection<ImageFrame>* dImageFrame
      = new PipeDataInCollection<ImageFrame>(device_count);
  PipeDataInCollectionOnce<CalibData>* dCalibData
      = new PipeDataInCollectionOnce<CalibData>(device_count);

  std::vector<std::string> serial_nums(device_count);
  KinectCameras* cameras = new KinectCameras(
      dImageFrame,
      dCalibData,
      &serial_nums,
      jsonConf["kinect_config"]["device_count"].asInt(),
      jsonConf["kinect_config"]["master_serial"].asString(),
      jsonConf["kinect_config"]["camera_resolution"]["width"].asInt(),
      jsonConf["kinect_config"]["camera_resolution"]["height"].asInt(),
      jsonConf["device_calibration"],
      true);

  SenderToPython* sender = new SenderToPython("127.0.0.1", 33669);
  std::thread consumer_thread(&image_consumer, dImageFrame, sender);

  for (auto& thread : *cameras->get_cam_threads())
  {
    thread.join();
  }

  consumer_thread.join();

  std::cout << "finish" << std::endl;

  return 0;
}
