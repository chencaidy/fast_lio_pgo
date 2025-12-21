#include "common.h"
#include <sensor_msgs/msg/point_cloud2.hpp>

// namespace fast_lio {

enum LidarModel {
  UNDIS = 0,
  SEYOND,
  HESAI,
};

class LaserProcess {
public:
  LaserProcess(LidarModel type);
  ~LaserProcess();

  void set_crop_size(double size);

  void process(const sensor_msgs::msg::PointCloud2::UniquePtr &msg, PointCloud::Ptr &output);

private:
  PointCloud::Ptr default_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg);

  LidarModel lidar_model_;
  double filter_crop_ = 0;
};

// } // namespace fast_lio
