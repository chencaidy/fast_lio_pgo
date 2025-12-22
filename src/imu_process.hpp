#pragma once

#include "common.h"
#include "esekfom.hpp"

#include <Eigen/Core>
#include <fstream>

#include <fast_lio_pgo/msg/pose6_d.hpp>
#include <sensor_msgs/msg/imu.hpp>

class ImuProcess {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();

  void reset();

  void set_extrinsic(const Eigen::Matrix4d &transform);
  void set_extrinsic(const Eigen::Vector3d &T, const Eigen::Matrix3d &R);
  void set_extrinsic(const Eigen::Vector3d &T);

  void set_acc_cov(const Eigen::Vector3d &scale);
  void set_gyr_cov(const Eigen::Vector3d &scale);
  void set_acc_bias_cov(const Eigen::Vector3d &bias);
  void set_gyr_bias_cov(const Eigen::Vector3d &bias);

  void process(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloud::Ptr pcl_un_);

private:
  void imu_init(const MeasureGroup &meas, esekfom::esekf &kf_state, int &N);
  void undistort_pcl(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloud &pcl_in_out);

  Eigen::Matrix<double, 12, 12> process_noise_cov();

  template <typename T>
  fast_lio_pgo::msg::Pose6D to_pose6d(const double t, const Eigen::Matrix<T, 3, 1> &a, const Eigen::Matrix<T, 3, 1> &g,
                                      const Eigen::Matrix<T, 3, 1> &v, const Eigen::Matrix<T, 3, 1> &p,
                                      const Eigen::Matrix<T, 3, 3> &R);

private:
  bool imu_need_init_ = true;
  int imu_init_cnt_ = 1;
  std::vector<fast_lio_pgo::msg::Pose6D> imu_pose_;
  sensor_msgs::msg::Imu::ConstSharedPtr imu_last_;

  bool b_first_frame_ = true;
  double last_lidar_end_time_ = 0;
  Eigen::Matrix3d Lidar_R_wrt_IMU;
  Eigen::Vector3d Lidar_T_wrt_IMU;

  Eigen::Vector3d mean_acc_;
  Eigen::Vector3d mean_gyr_;
  Eigen::Vector3d last_acc_;
  Eigen::Vector3d last_gyr_;

  Eigen::Matrix<double, 12, 12> Q_;
  Eigen::Vector3d cov_acc_;
  Eigen::Vector3d cov_gyr_;
  Eigen::Vector3d cov_acc_scale_;
  Eigen::Vector3d cov_gyr_scale_;
  Eigen::Vector3d cov_acc_bias_;
  Eigen::Vector3d cov_gyr_bias_;

  std::ofstream fout_imu_;
};
