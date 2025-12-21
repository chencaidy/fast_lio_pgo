#pragma once

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/msg/imu.hpp>

#include <deque>
#include <vector>

#define G_m_s2 (9.80665) // Gravity constant
#define NUM_MATCH_POINTS (5)
#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]
#define MAT_FROM_ARRAY(v) v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8]
#define DEBUG_FILE_DIR(name) (std::string(std::string(ROOT_DIR) + "/Log/" + name))

struct EIGEN_ALIGN16 PointXYZIT {
  PCL_ADD_POINT4D;
  float intensity;
  double timestamp;
  PCL_MAKE_ALIGNED_OPERATOR_NEW
};

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloud;
typedef std::vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;

struct MeasureGroup {
  double lidar_beg_time;
  double lidar_end_time;
  PointCloud::Ptr lidar;
  std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu;

  MeasureGroup() {
    lidar_beg_time = 0.0;
    lidar_end_time = 0.0;
    this->lidar.reset(new PointCloud());
  };
};

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(PointXYZIT,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, intensity, intensity)
  (double, timestamp, timestamp)
)
// clang-format on
