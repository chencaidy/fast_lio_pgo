#include "laser_process.hpp"
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>

// namespace fast_lio {

LaserProcess::LaserProcess(LidarModel type) { lidar_model_ = type; }

LaserProcess::~LaserProcess() {}

void LaserProcess::process(const sensor_msgs::msg::PointCloud2::UniquePtr &msg, PointCloud::Ptr &output) {
  switch (lidar_model_) {
  case UNDIS:
    output = default_handler(msg);
    break;
  default:
    break;
  }
}

void LaserProcess::set_crop_size(double size) { filter_crop_ = size; }

PointCloud::Ptr LaserProcess::default_handler(const sensor_msgs::msg::PointCloud2::UniquePtr &msg) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr original_scan(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::fromROSMsg(*msg, *original_scan);

  pcl::Indices index;
  pcl::removeNaNFromPointCloud(*original_scan, *original_scan, index);

  pcl::CropBox<pcl::PointXYZI> crop;
  crop.setMin(Eigen::Vector4f(-filter_crop_, -filter_crop_, -filter_crop_, 1.0));
  crop.setMax(Eigen::Vector4f(filter_crop_, filter_crop_, filter_crop_, 1.0));
  crop.setNegative(true);
  crop.setInputCloud(original_scan);
  crop.filter(*original_scan);

  PointCloud::Ptr filtered_scan(new PointCloud);
  for (const auto &point : *original_scan) {
    PointType target;
    target.x = point.x;
    target.y = point.y;
    target.z = point.z;
    target.intensity = point.intensity;
    target.curvature = 0;
    filtered_scan->push_back(target);
  }

  return filtered_scan;
}

// } // namespace fast_lio
