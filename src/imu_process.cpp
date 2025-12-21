#include "imu_process.hpp"
#include "so3_math.h"

#include <omp.h>
#include <rclcpp/rclcpp.hpp>
#include <string>

#define LOGGER rclcpp::get_logger("laser_mapping")
#define MAX_INIT_COUNT (10)

bool time_list(PointType &x, PointType &y) { return (x.curvature < y.curvature); };

ImuProcess::ImuProcess() {
  Lidar_T_wrt_IMU = Eigen::Vector3d(0, 0, 0);
  Lidar_R_wrt_IMU = Eigen::Matrix3d::Identity();

  Q_ = process_noise_cov();
  cov_acc_scale_ = Eigen::Vector3d(0.1, 0.1, 0.1);
  cov_gyr_scale_ = Eigen::Vector3d(0.1, 0.1, 0.1);
  cov_acc_bias_ = Eigen::Vector3d(0.0001, 0.0001, 0.0001);
  cov_gyr_bias_ = Eigen::Vector3d(0.0001, 0.0001, 0.0001);

  mean_acc_ = Eigen::Vector3d(0, 0, -1.0);
  mean_gyr_ = Eigen::Vector3d(0, 0, 0);
  last_acc_ = Eigen::Vector3d(0, 0, 0);
  last_gyr_ = Eigen::Vector3d(0, 0, 0);

  imu_pose_.clear();
  imu_last_.reset(new sensor_msgs::msg::Imu());
}

ImuProcess::~ImuProcess() {}

void ImuProcess::reset() {
  mean_acc_ = Eigen::Vector3d(0, 0, -1.0);
  mean_gyr_ = Eigen::Vector3d(0, 0, 0);
  last_acc_ = Eigen::Vector3d(0, 0, 0);
  last_gyr_ = Eigen::Vector3d(0, 0, 0);

  imu_need_init_ = true;
  imu_init_cnt_ = 1;
  imu_pose_.clear();
  imu_last_.reset(new sensor_msgs::msg::Imu());

  RCLCPP_WARN(LOGGER, "IMU reset done");
}

void ImuProcess::set_extrinsic(const Eigen::Matrix4d &transform) {
  Lidar_T_wrt_IMU = transform.block<3, 1>(0, 3);
  Lidar_R_wrt_IMU = transform.block<3, 3>(0, 0);
}

void ImuProcess::set_extrinsic(const Eigen::Vector3d &T, const Eigen::Matrix3d &R) {
  Lidar_T_wrt_IMU = T;
  Lidar_R_wrt_IMU = R;
}

void ImuProcess::set_extrinsic(const Eigen::Vector3d &T) {
  Lidar_T_wrt_IMU = T;
  Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_acc_cov(const Eigen::Vector3d &scale) { cov_acc_scale_ = scale; }

void ImuProcess::set_gyr_cov(const Eigen::Vector3d &scale) { cov_gyr_scale_ = scale; }

void ImuProcess::set_acc_bias_cov(const Eigen::Vector3d &bias) { cov_acc_bias_ = bias; }

void ImuProcess::set_gyr_bias_cov(const Eigen::Vector3d &bias) { cov_gyr_bias_ = bias; }

void ImuProcess::process(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloud::Ptr pc) {
  double t1 = omp_get_wtime();

  if (meas.imu.empty()) {
    return;
  };
  assert(meas.lidar != nullptr);

  if (imu_need_init_) {
    imu_init(meas, kf_state, imu_init_cnt_);
    imu_last_ = meas.imu.back();

    esekfom::state_ikfom imu_state = kf_state.get_x();
    if (imu_init_cnt_ > MAX_INIT_COUNT) {
      imu_need_init_ = false;

      // cov_acc *= pow(G_m_s2 / mean_acc_.norm(), 2);
      cov_acc_ = cov_acc_scale_;
      cov_gyr_ = cov_gyr_scale_;

      RCLCPP_INFO(LOGGER, "IMU initial done");
      RCLCPP_INFO(LOGGER, "Gravity: %f %f %f %f", imu_state.grav[0], imu_state.grav[1], imu_state.grav[2],
                  mean_acc_.norm());
      RCLCPP_INFO(LOGGER, "Accel covarience: %f %f %f", cov_acc_[0], cov_acc_[1], cov_acc_[2]);
      RCLCPP_INFO(LOGGER, "Gyro covarience: %f %f %f", cov_gyr_[0], cov_gyr_[1], cov_gyr_[2]);
      RCLCPP_INFO(LOGGER, "Gyro bias covarience: %f %f %f", cov_gyr_bias_[0], cov_gyr_bias_[1], cov_gyr_bias_[2]);

      fout_imu_.open(DEBUG_FILE_DIR("imu.txt"), std::ios::out);
    }

    return;
  }

  undistort_pcl(meas, kf_state, *pc);

  double t2 = omp_get_wtime();
  RCLCPP_DEBUG(LOGGER, "IMU processing time: %fs", t2 - t1);
}

// 1. Initializing the gravity, gyro bias, acc and gyro covariance.
// 2. Normalize the acceleration measurements to unit gravity.
void ImuProcess::imu_init(const MeasureGroup &meas, esekfom::esekf &kf_state, int &N) {
  Eigen::Vector3d cur_acc, cur_gyr;

  if (b_first_frame_) {
    reset();
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc_ << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr_ << gyr_acc.x, gyr_acc.y, gyr_acc.z;
  }

  for (const auto &imu : meas.imu) {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &imu_gyr = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << imu_gyr.x, imu_gyr.y, imu_gyr.z;

    mean_acc_ += (cur_acc - mean_acc_) / N;
    mean_gyr_ += (cur_gyr - mean_gyr_) / N;

    cov_acc_ = cov_acc_ * (N - 1.0) / N + (cur_acc - mean_acc_).cwiseProduct(cur_acc - mean_acc_) * (N - 1.0) / (N * N);
    cov_gyr_ = cov_gyr_ * (N - 1.0) / N + (cur_gyr - mean_gyr_).cwiseProduct(cur_gyr - mean_gyr_) * (N - 1.0) / (N * N);

    // RCLCPP_INFO(LOGGER, "acc norm: %f %f", cur_acc.norm(), mean_acc_.norm());

    N++;
  }

  esekfom::state_ikfom init_state = kf_state.get_x();
  init_state.grav = -mean_acc_ / mean_acc_.norm() * G_m_s2;
  init_state.bg = mean_gyr_;
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;
  kf_state.change_x(init_state);

  Eigen::Matrix<double, 24, 24> init_P = Eigen::MatrixXd::Identity(24, 24);
  init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;
  init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;
  init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;
  init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;
  init_P(21, 21) = init_P(22, 22) = init_P(23, 23) = 0.00001;
  kf_state.change_P(init_P);
  imu_last_ = meas.imu.back();
}

void ImuProcess::undistort_pcl(const MeasureGroup &meas, esekfom::esekf &kf_state, PointCloud &pcl_out) {
  /* Add the imu of the last frame-tail to the of current frame-head */
  auto v_imu = meas.imu;
  v_imu.push_front(imu_last_);
  const double &imu_beg_time = rclcpp::Time(v_imu.front()->header.stamp).seconds();
  const double &imu_end_time = rclcpp::Time(v_imu.back()->header.stamp).seconds();
  const double &pcl_beg_time = meas.lidar_beg_time;
  const double &pcl_end_time = meas.lidar_end_time;

  /* Sort pointclouds by offset time */
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  RCLCPP_DEBUG(LOGGER, "Process lidar from %f to %f", pcl_beg_time, pcl_end_time);
  RCLCPP_DEBUG(LOGGER, "Process %ld IMU msgs from from %f to %f", meas.imu.size(), imu_beg_time, imu_end_time);

  /* Initialize IMU pose */
  esekfom::state_ikfom imu_state = kf_state.get_x();
  auto pose6d = to_pose6d(0, last_acc_, last_gyr_, imu_state.vel, imu_state.pos, imu_state.rot.matrix());
  imu_pose_.clear();
  imu_pose_.push_back(pose6d);

  /* Forward propagation at each imu point */
  Eigen::Vector3d angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  Eigen::Matrix3d R_imu;
  double dt = 0;
  esekfom::input_ikfom imu_input;
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);

    double tail_stamp = rclcpp::Time(tail->header.stamp).seconds();
    double head_stamp = rclcpp::Time(head->header.stamp).seconds();

    if (tail_stamp < last_lidar_end_time_)
      continue;

    angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
        0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
        0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
        0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
        0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // fout_imu_ << head_stamp << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << std::endl;

    acc_avr = acc_avr * G_m_s2 / mean_acc_.norm();

    if (head_stamp < last_lidar_end_time_) {
      dt = tail_stamp - last_lidar_end_time_;
    } else {
      dt = tail_stamp - head_stamp;
    }

    imu_input.acc = acc_avr;
    imu_input.gyro = angvel_avr;
    Q_.block<3, 3>(0, 0).diagonal() = cov_gyr_;
    Q_.block<3, 3>(3, 3).diagonal() = cov_acc_;
    Q_.block<3, 3>(6, 6).diagonal() = cov_gyr_bias_;
    Q_.block<3, 3>(9, 9).diagonal() = cov_acc_bias_;
    kf_state.predict(dt, Q_, imu_input);

    /* Save the poses at each IMU measurements */
    imu_state = kf_state.get_x();
    last_gyr_ = angvel_avr - imu_state.bg;
    last_acc_ = imu_state.rot * (acc_avr - imu_state.ba);
    for (int i = 0; i < 3; i++) {
      last_acc_[i] += imu_state.grav[i];
    }
    double offs_t = tail_stamp - pcl_beg_time;
    auto pose6d = to_pose6d(offs_t, last_acc_, last_gyr_, imu_state.vel, imu_state.pos, imu_state.rot.matrix());
    imu_pose_.push_back(pose6d);
  }

  /* Calculated the pos and attitude prediction at the frame-end */
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q_, imu_input);

  imu_state = kf_state.get_x();
  imu_last_ = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

  /* Undistort each lidar point (backward propagation) */
  if (pcl_out.points.begin() == pcl_out.points.end())
    return;
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = imu_pose_.end() - 1; it_kp != imu_pose_.begin(); it_kp--) {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu << MAT_FROM_ARRAY(head->rot);
    // RCLCPP_INFO_STREAM(LOGGER, "head imu acc: " << acc_imu.transpose());
    vel_imu << VEC_FROM_ARRAY(head->vel);
    pos_imu << VEC_FROM_ARRAY(head->pos);
    acc_imu << VEC_FROM_ARRAY(tail->acc);
    angvel_avr << VEC_FROM_ARRAY(tail->gyr);

    for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      Eigen::Matrix3d R_i(R_imu * Exp(angvel_avr, dt));

      Eigen::Vector3d P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      Eigen::Vector3d T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
      Eigen::Vector3d P_compensate =
          imu_state.offset_R_L_I.matrix().transpose() *
          (imu_state.rot.matrix().transpose() *
               (R_i * (imu_state.offset_R_L_I.matrix() * P_i + imu_state.offset_T_L_I) + T_ei) -
           imu_state.offset_T_L_I); // Not accurate!

      // Save undistorted points and rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin())
        break;
    }
  }
}

Eigen::Matrix<double, 12, 12> ImuProcess::process_noise_cov() {
  Eigen::Matrix<double, 12, 12> Q = Eigen::MatrixXd::Zero(12, 12);
  Q.block<3, 3>(0, 0) = 0.0001 * Eigen::Matrix3d::Identity();
  Q.block<3, 3>(3, 3) = 0.0001 * Eigen::Matrix3d::Identity();
  Q.block<3, 3>(6, 6) = 0.00001 * Eigen::Matrix3d::Identity();
  Q.block<3, 3>(9, 9) = 0.00001 * Eigen::Matrix3d::Identity();
  return Q;
}

template <typename T>
fast_lio::msg::Pose6D ImuProcess::to_pose6d(const double t, const Eigen::Matrix<T, 3, 1> &a,
                                            const Eigen::Matrix<T, 3, 1> &g, const Eigen::Matrix<T, 3, 1> &v,
                                            const Eigen::Matrix<T, 3, 1> &p, const Eigen::Matrix<T, 3, 3> &R) {
  fast_lio::msg::Pose6D rot_kp;
  rot_kp.offset_time = t;
  for (int i = 0; i < 3; i++) {
    rot_kp.acc[i] = a(i);
    rot_kp.gyr[i] = g(i);
    rot_kp.vel[i] = v(i);
    rot_kp.pos[i] = p(i);
    for (int j = 0; j < 3; j++)
      rot_kp.rot[i * 3 + j] = R(i, j);
  }
  return rot_kp;
}
