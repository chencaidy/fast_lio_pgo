// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "imu_process.hpp"
#include "laser_process.hpp"

#include <omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <chrono>
#include <mutex>
#include <string>

#define INIT_TIME 0.1
#define LASER_POINT_COV 0.001
#define MOV_THRESHOLD 1.5

using namespace std::chrono_literals;

inline float calc_dist(PointType p1, PointType p2) {
  float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
  return d;
}

inline double get_time_sec(const builtin_interfaces::msg::Time &time) { return rclcpp::Time(time).seconds(); }

inline rclcpp::Time get_ros_time(double timestamp) {
  int32_t sec = std::floor(timestamp);
  auto nanosec_d = (timestamp - std::floor(timestamp)) * 1e9;
  uint32_t nanosec = nanosec_d;
  return rclcpp::Time(sec, nanosec);
}

class LaserMappingNode : public rclcpp::Node {
public:
  LaserMappingNode(const rclcpp::NodeOptions &options) : Node("laser_mapping", options) {

    declare_parameter<std::string>("common.lidar_topic", "/points");
    declare_parameter<std::string>("common.imu_topic", "/imu");
    declare_parameter<std::string>("common.initial_pose_topic", "/initialpose");
    declare_parameter<std::string>("common.lidar_frame_id", "lidar");
    declare_parameter<std::string>("common.imu_frame_id", "imu");
    declare_parameter<double>("common.time_offset_lidar_to_imu", 0);

    declare_parameter<int>("preprocess.lidar_type", UNDIS);
    declare_parameter<double>("preprocess.blind", 0);

    declare_parameter<double>("mapping.gyr_cov", 0.1);
    declare_parameter<double>("mapping.acc_cov", 0.1);
    declare_parameter<double>("mapping.b_gyr_cov", 0.0001);
    declare_parameter<double>("mapping.b_acc_cov", 0.0001);
    declare_parameter<double>("mapping.cube_side_length", 200);
    declare_parameter<double>("mapping.det_range", 300);
    declare_parameter<int>("mapping.max_iteration", 4);
    declare_parameter<bool>("mapping.extrinsic_est_en", true);
    declare_parameter<double>("mapping.filter_size_surf", 0.5);
    declare_parameter<double>("mapping.filter_size_map", 0.5);

    declare_parameter<bool>("publish.path_en", true);
    declare_parameter<bool>("publish.scan_publish_en", true);
    declare_parameter<bool>("publish.scan_bodyframe_pub_en", true);
    declare_parameter<bool>("publish.dense_publish_en", true);
    // declare_parameter<bool>("publish.effect_map_en", false);
    declare_parameter<bool>("publish.map_en", false);

    std::string lidar_topic, imu_topic, initial_pose_topic;
    get_parameter<std::string>("common.lidar_topic", lidar_topic);
    get_parameter<std::string>("common.imu_topic", imu_topic);
    get_parameter<std::string>("common.initial_pose_topic", initial_pose_topic);
    get_parameter<std::string>("common.lidar_frame_id", lidar_frame_id);
    get_parameter<std::string>("common.imu_frame_id", imu_frame_id);
    get_parameter<double>("common.time_offset_lidar_to_imu", time_offset_lidar_to_imu);

    int lidar_type;
    double crop_size;
    get_parameter<int>("preprocess.lidar_type", lidar_type);
    get_parameter<double>("preprocess.blind", crop_size);

    double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
    get_parameter<double>("mapping.gyr_cov", gyr_cov);
    get_parameter<double>("mapping.acc_cov", acc_cov);
    get_parameter<double>("mapping.b_gyr_cov", b_gyr_cov);
    get_parameter<double>("mapping.b_acc_cov", b_acc_cov);
    get_parameter<double>("mapping.cube_side_length", cube_len);
    get_parameter<double>("mapping.det_range", DET_RANGE);
    get_parameter<int>("mapping.max_iteration", max_iteration);
    get_parameter<bool>("mapping.extrinsic_est_en", extrinsic_est_en);
    get_parameter<double>("mapping.filter_size_surf", filter_size_surf_min);
    get_parameter<double>("mapping.filter_size_map", filter_size_map_min);

    get_parameter<bool>("publish.path_en", path_en);
    get_parameter<bool>("publish.scan_publish_en", scan_pub_en);
    get_parameter<bool>("publish.scan_bodyframe_pub_en", scan_body_pub_en);
    get_parameter<bool>("publish.dense_publish_en", dense_pub_en);
    // get_parameter<bool>("publish.effect_map_en", effect_pub_en);
    get_parameter<bool>("publish.map_en", map_pub_en);

    // Get transform
    Eigen::Vector3d T_lidar_to_imu;
    Eigen::Matrix3d R_lidar_to_imu;
    tf2_ros::Buffer tf_buffer(get_clock());
    tf2_ros::TransformListener tf_listener(tf_buffer);
    try {
      auto tf_lidar_to_imu = tf_buffer.lookupTransform(imu_frame_id, lidar_frame_id, tf2::TimePointZero, 5s);
      // Translation
      auto translation = tf_lidar_to_imu.transform.translation;
      T_lidar_to_imu = Eigen::Vector3d(translation.x, translation.y, translation.z);
      // Rotation
      auto rotation = tf_lidar_to_imu.transform.rotation;
      Eigen::Quaterniond quat(rotation.w, rotation.x, rotation.y, rotation.z);
      quat.normalize();
      R_lidar_to_imu = quat.toRotationMatrix();
      // Print result
      RCLCPP_INFO_STREAM(get_logger(), "Transform from " << imu_frame_id << " to " << lidar_frame_id);
      RCLCPP_DEBUG_STREAM(get_logger(), "T: " << T_lidar_to_imu);
      RCLCPP_DEBUG_STREAM(get_logger(), "R: " << R_lidar_to_imu);
    } catch (const tf2::TransformException &ex) {
      RCLCPP_ERROR(get_logger(), "Could not lookup transform: %s", ex.what());
      throw;
    }

    p_laser_ = std::make_shared<LaserProcess>((LidarModel)lidar_type);
    p_laser_->set_crop_size(crop_size);

    p_imu_ = std::make_shared<ImuProcess>();
    p_imu_->set_extrinsic(T_lidar_to_imu, R_lidar_to_imu);
    p_imu_->set_gyr_cov(Eigen::Vector3d(gyr_cov, gyr_cov, gyr_cov));
    p_imu_->set_acc_cov(Eigen::Vector3d(acc_cov, acc_cov, acc_cov));
    p_imu_->set_gyr_bias_cov(Eigen::Vector3d(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu_->set_acc_bias_cov(Eigen::Vector3d(b_acc_cov, b_acc_cov, b_acc_cov));

    voxel_filter_surf_.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    voxel_filter_map_.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    ikdtree.set_downsample_param(filter_size_map_min);

    feats_from_map = std::make_shared<PointCloud>();
    feats_undistort = std::make_shared<PointCloud>();
    feats_down_body = std::make_shared<PointCloud>();
    feats_down_world = std::make_shared<PointCloud>();
    feats_deleted = std::make_shared<PointCloud>();
    pcl_wait_pub = std::make_shared<PointCloud>();

    // Debug record
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.csv"), std::ios::out);
    if (fout_pre) {
      RCLCPP_INFO(get_logger(), "Log file save to %s", ROOT_DIR);
    } else {
      RCLCPP_ERROR(get_logger(), "Path %s doesn't exist", ROOT_DIR);
    }
    fout_pre << "time,"
             << "roll,pitch,yaw,"
             << "position.x,position.y,position.z,"
             << "extrinsics.r,extrinsics.p,extrinsics.y,"
             << "extrinsics.x,extrinsics.y,extrinsics.z,"
             << "velocity.x,velocity.y,velocity.z,"
             << "bias_gyro.x,bias_gyro.y,bias_gyro.z,"
             << "bias_accel.x,bias_accel.y,bias_accel.z,"
             << "gravity.x,gravity.y,gravity.z" << std::endl;

    /*** ROS subscribe initialization ***/
    sub_lidar_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        lidar_topic, rclcpp::SensorDataQoS(),
        std::bind(&LaserMappingNode::standard_pcl_cbk, this, std::placeholders::_1));
    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
        imu_topic, rclcpp::SensorDataQoS(), std::bind(&LaserMappingNode::imu_cbk, this, std::placeholders::_1));
    sub_pose_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        initial_pose_topic, rclcpp::SensorDataQoS(),
        std::bind(&LaserMappingNode::pose_cbk, this, std::placeholders::_1));

    pub_frame_map_ = create_publisher<sensor_msgs::msg::PointCloud2>("/mapping/cloud_registered", 20);
    pub_frame_body_ = create_publisher<sensor_msgs::msg::PointCloud2>("/mapping/cloud_registered_body", 20);
    // pub_effect_ = create_publisher<sensor_msgs::msg::PointCloud2>("/mapping/cloud_effected", 20);
    pub_map_ = create_publisher<sensor_msgs::msg::PointCloud2>("/mapping/laser_map", 20);
    pub_odom_ = create_publisher<nav_msgs::msg::Odometry>("/mapping/odometry", 20);
    pub_path_ = create_publisher<nav_msgs::msg::Path>("/mapping/path", 20);

    tf_pub_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    timer_ = rclcpp::create_timer(this, get_clock(), 10ms, std::bind(&LaserMappingNode::timer_callback, this));
    map_pub_timer_ =
        rclcpp::create_timer(this, get_clock(), 1s, std::bind(&LaserMappingNode::map_publish_callback, this));

    RCLCPP_INFO(this->get_logger(), "Node init finished.");
  }

  ~LaserMappingNode() { fout_pre.close(); }

  void save_log_state_to_csv(double time, const esekfom::state_ikfom &state) {
    Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", ",", "", "", "", "");
    Eigen::Vector3d rot_euler = so3_to_euler(state.rot);
    Eigen::Vector3d ext_euler = so3_to_euler(state.offset_R_L_I);

    // clang-format off
    fout_pre << time << ","
             << rot_euler.transpose().format(fmt) << ","
             << state.pos.transpose().format(fmt) << ","
             << ext_euler.transpose().format(fmt) << ","
             << state.offset_T_L_I.transpose().format(fmt) << ","
             << state.vel.transpose().format(fmt) << ","
             << state.bg.transpose().format(fmt) << ","
             << state.ba.transpose().format(fmt) << ","
             << state.grav[0] << "," << state.grav[1] << "," << state.grav[2]
             << std::endl;
    // clang-format on
  }

private:
  void timer_callback() {
    if (sync_packages(measures)) {
      if (std::fabs(first_lidar_time) < 1e-15) {
        first_lidar_time = measures.lidar_beg_time;
        return;
      }

      double t_begin = omp_get_wtime();

      p_imu_->process(measures, kf, feats_undistort);
      if (feats_undistort->empty() || (feats_undistort == NULL)) {
        RCLCPP_WARN(this->get_logger(), "No point, skip this scan!");
        return;
      }

      double t_undis = omp_get_wtime();

      /*** Segment the map in lidar FOV ***/
      lasermap_fov_segment();

      double t_segment = omp_get_wtime();

      /*** downsample the feature points in a scan ***/
      voxel_filter_surf_.setInputCloud(feats_undistort);
      voxel_filter_surf_.filter(*feats_down_body);

      double t_downsample = omp_get_wtime();

      /*** Initialize the map kdtree ***/
      feats_down_size = feats_down_body->points.size();

      if (ikdtree.Root_Node == nullptr) {
        RCLCPP_INFO(this->get_logger(), "Initialize the map kdtree");
        if (feats_down_size > 5) {
          feats_down_world->resize(feats_down_size);
          for (int i = 0; i < feats_down_size; i++) {
            point_body_to_world(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
          }
          ikdtree.Build(feats_down_world->points);
        }
        return;
      }

      RCLCPP_DEBUG(get_logger(), "Input: %ld Downsample: %d Map: %d", feats_undistort->points.size(), feats_down_size,
                   ikdtree.validnum());

      /*** ICP and iterated Kalman filter update ***/
      if (feats_down_size < 5) {
        RCLCPP_WARN(this->get_logger(), "No point, skip this scan!");
        return;
      }

#if 0 // If you need to see map point, change to "if(1)"
        PointVector().swap(ikdtree.PCL_Storage);
        ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
        feats_from_map->clear();
        feats_from_map->points = ikdtree.PCL_Storage;
#endif

      /*** Iterated state estimation ***/
      nearest_points.resize(feats_down_size);
      kf.update_iterated_dyn_share_modified(LASER_POINT_COV, feats_down_body, ikdtree, nearest_points, max_iteration,
                                            extrinsic_est_en);
      state_point = kf.get_x();

      double t_icp = omp_get_wtime();

      /*** Add the feature points to map kdtree ***/
      feats_down_world->resize(feats_down_size);
      map_incremental();

      double t_submap = omp_get_wtime();

      /*** Publish odometry ***/
      publish_odometry(state_point);
      if (path_en) {
        publish_path(state_point);
      }

      /*** Publish points ***/
      if (scan_pub_en) {
        publish_frame_world(pub_frame_map_);
      }
      if (scan_pub_en && scan_body_pub_en) {
        publish_frame_body(pub_frame_body_);
      }
      // if (effect_pub_en) {
      //   publish_effect_world(pub_effect_);
      // }

      double t_publish = omp_get_wtime();

      /*** Debug variables ***/
      save_log_state_to_csv(measures.lidar_beg_time - first_lidar_time, state_point);

      RCLCPP_INFO(get_logger(),
                  "[Perf] Undis: %0.6f Segment: %0.6f Downsample: %0.6f ICP: %0.6f Submap: %0.6f Total: %0.6f",
                  t_undis - t_begin, t_segment - t_undis, t_downsample - t_segment, t_icp - t_downsample,
                  t_submap - t_icp, t_publish - t_begin);
    }
  }

  void map_publish_callback() {
    if (map_pub_en)
      publish_map(pub_map_);
  }

  bool sync_packages(MeasureGroup &meas) {
    static int lidar_scan_num = 0;
    static double lidar_mean_scantime = 0;

    std::lock_guard<std::mutex> lock(mtx_buffer);

    if (lidar_buffer.empty() || imu_buffer.empty()) {
      return false;
    }

    /*** push a lidar scan ***/
    if (time_buffer.front() > meas.lidar_beg_time) {
      meas.lidar = lidar_buffer.front();
      meas.lidar_beg_time = time_buffer.front();
      if (meas.lidar->points.size() <= 1) // time too little
      {
        lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        RCLCPP_ERROR(get_logger(), "Too few input point cloud!");
      } else if (meas.lidar->points.back().curvature / 1000.0 < 0.5 * lidar_mean_scantime) {
        lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
      } else {
        lidar_scan_num++;
        lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / 1000.0;
        lidar_mean_scantime += (meas.lidar->points.back().curvature / 1000.0 - lidar_mean_scantime) / lidar_scan_num;
      }
      meas.lidar_end_time = lidar_end_time;
    }

    if (last_timestamp_imu < lidar_end_time) {
      return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = get_time_sec(imu_buffer.front()->header.stamp);
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time)) {
      imu_time = get_time_sec(imu_buffer.front()->header.stamp);
      if (imu_time > lidar_end_time)
        break;
      meas.imu.push_back(imu_buffer.front());
      imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();

    return true;
  }

  void publish_odometry(const esekfom::state_ikfom &state) {
    nav_msgs::msg::Odometry odom;
    odom.header.stamp = get_ros_time(lidar_end_time);
    odom.header.frame_id = "map";
    odom.child_frame_id = imu_frame_id;
    odom.pose.pose.position.x = state.pos(0);
    odom.pose.pose.position.y = state.pos(1);
    odom.pose.pose.position.z = state.pos(2);
    auto quat = Eigen::Quaterniond(state.rot.matrix());
    odom.pose.pose.orientation.x = quat.coeffs()[0];
    odom.pose.pose.orientation.y = quat.coeffs()[1];
    odom.pose.pose.orientation.z = quat.coeffs()[2];
    odom.pose.pose.orientation.w = quat.coeffs()[3];
    auto P = kf.get_P();
    for (int i = 0; i < 6; i++) {
      int k = i < 3 ? i + 3 : i - 3;
      odom.pose.covariance[i * 6 + 0] = P(k, 3);
      odom.pose.covariance[i * 6 + 1] = P(k, 4);
      odom.pose.covariance[i * 6 + 2] = P(k, 5);
      odom.pose.covariance[i * 6 + 3] = P(k, 0);
      odom.pose.covariance[i * 6 + 4] = P(k, 1);
      odom.pose.covariance[i * 6 + 5] = P(k, 2);
    }
    pub_odom_->publish(odom);

    geometry_msgs::msg::TransformStamped trans;
    trans.header.frame_id = "map";
    trans.child_frame_id = imu_frame_id;
    trans.header.stamp = get_ros_time(lidar_end_time);
    trans.transform.translation.x = odom.pose.pose.position.x;
    trans.transform.translation.y = odom.pose.pose.position.y;
    trans.transform.translation.z = odom.pose.pose.position.z;
    trans.transform.rotation = odom.pose.pose.orientation;
    tf_pub_->sendTransform(trans);
  }

  void publish_path(const esekfom::state_ikfom &state) {
    static int path_cnt = 0;
    if (path_cnt % 10 == 0) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header.stamp = get_ros_time(lidar_end_time);
      pose.header.frame_id = "map";
      pose.pose.position.x = state.pos(0);
      pose.pose.position.y = state.pos(1);
      pose.pose.position.z = state.pos(2);
      auto quat = Eigen::Quaterniond(state.rot.matrix());
      pose.pose.orientation.x = quat.coeffs()[0];
      pose.pose.orientation.y = quat.coeffs()[1];
      pose.pose.orientation.z = quat.coeffs()[2];
      pose.pose.orientation.w = quat.coeffs()[3];
      path.header.stamp = now();
      path.header.frame_id = "map";
      path.poses.push_back(pose);
      pub_path_->publish(path);
    }
    path_cnt++;
  }

  // void publish_effect_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect) {
  //   PointCloud::Ptr laserCloudWorld(new PointCloud(effct_feat_num, 1));
  //   for (int i = 0; i < effct_feat_num; i++) {
  //     point_body_to_world(&laserCloudOri->points[i], &laserCloudWorld->points[i]);
  //   }
  //   sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
  //   pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
  //   laserCloudFullRes3.header.stamp = get_ros_time(lidar_end_time);
  //   laserCloudFullRes3.header.frame_id = "map";
  //   pubLaserCloudEffect->publish(laserCloudFullRes3);
  // }

  void publish_map(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap) {
    PointCloud::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloud::Ptr laserCloudWorld(new PointCloud(size, 1));

    for (int i = 0; i < size; i++) {
      point_body_to_world(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
    }
    *pcl_wait_pub += *laserCloudWorld;

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*pcl_wait_pub, msg);
    msg.header.stamp = get_ros_time(lidar_end_time);
    msg.header.frame_id = "map";
    pubLaserCloudMap->publish(msg);

    // sensor_msgs::msg::PointCloud2 msg;
    // pcl::toROSMsg(*feats_from_map, msg);
    // msg.header.stamp = get_ros_time(lidar_end_time);
    // msg.header.frame_id = "map";
    // pubLaserCloudMap->publish(msg);
  }

  void publish_frame_body(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull_body) {
    int size = feats_undistort->points.size();
    PointCloud::Ptr laserCloudIMUBody(new PointCloud(size, 1));

    for (int i = 0; i < size; i++) {
      point_body_lidar_to_imu(&feats_undistort->points[i], &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = imu_frame_id;
    pubLaserCloudFull_body->publish(laserCloudmsg);
  }

  void publish_frame_world(rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFull) {
    PointCloud::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloud::Ptr laserCloudWorld(new PointCloud(size, 1));

    for (int i = 0; i < size; i++) {
      point_body_to_world(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = "map";
    pubLaserCloudFull->publish(laserCloudmsg);
  }

  void map_incremental() {
    PointVector points_to_add;
    PointVector points_no_need_downsample;
    bool ekf_inited = (measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

    points_to_add.reserve(feats_down_size);
    points_no_need_downsample.reserve(feats_down_size);

    for (int i = 0; i < feats_down_size; i++) {
      /* transform to world frame */
      point_body_to_world(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
      /* decide if need add to map */
      if (!nearest_points[i].empty() && ekf_inited) {
        const PointVector &points_near = nearest_points[i];
        bool need_add = true;
        PointType downsample_result, mid_point;
        mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min +
                      0.5 * filter_size_map_min;
        mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min +
                      0.5 * filter_size_map_min;
        mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min +
                      0.5 * filter_size_map_min;
        float dist = calc_dist(feats_down_world->points[i], mid_point);
        if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min &&
            fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min &&
            fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min) {
          points_no_need_downsample.push_back(feats_down_world->points[i]);
          continue;
        }
        for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++) {
          if (points_near.size() < NUM_MATCH_POINTS)
            break;
          if (calc_dist(points_near[readd_i], mid_point) < dist) {
            need_add = false;
            break;
          }
        }
        if (need_add)
          points_to_add.push_back(feats_down_world->points[i]);
      } else {
        points_to_add.push_back(feats_down_world->points[i]);
      }
    }

    ikdtree.Add_Points(points_to_add, true);
    ikdtree.Add_Points(points_no_need_downsample, false);
    RCLCPP_DEBUG(get_logger(), "ikd-Tree add points: %ld", points_to_add.size() + points_no_need_downsample.size());
  }

  void lasermap_fov_segment() {
    std::vector<BoxPointType> cube_remove;
    esekfom::state_ikfom state = kf.get_x();
    Eigen::Vector3d pos_LiD = state.pos + state.rot * state.offset_T_L_I;

    if (!localmap_inited) {
      for (int i = 0; i < 3; i++) {
        localmap_points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
        localmap_points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
      }
      localmap_inited = true;
      return;
    }

    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++) {
      dist_to_map_edge[i][0] = fabs(pos_LiD(i) - localmap_points.vertex_min[i]);
      dist_to_map_edge[i][1] = fabs(pos_LiD(i) - localmap_points.vertex_max[i]);
      if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) {
        need_move = true;
      }
    }
    if (!need_move) {
      return;
    }

    BoxPointType new_localmap_points, tmp_boxpoints;
    new_localmap_points = localmap_points;
    float mov_dist =
        std::max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, (MOV_THRESHOLD - 1) * DET_RANGE);
    for (int i = 0; i < 3; i++) {
      tmp_boxpoints = localmap_points;
      if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE) {
        new_localmap_points.vertex_max[i] -= mov_dist;
        new_localmap_points.vertex_min[i] -= mov_dist;
        tmp_boxpoints.vertex_min[i] = localmap_points.vertex_max[i] - mov_dist;
        cube_remove.push_back(tmp_boxpoints);
      } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) {
        new_localmap_points.vertex_max[i] += mov_dist;
        new_localmap_points.vertex_min[i] += mov_dist;
        tmp_boxpoints.vertex_max[i] = localmap_points.vertex_min[i] + mov_dist;
        cube_remove.push_back(tmp_boxpoints);
      }
    }
    localmap_points = new_localmap_points;

    // PointVector points_history;
    // ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) {
    //   feats_deleted->push_back(points_history[i]);
    // }

    if (cube_remove.size() > 0) {
      // kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cube_remove);
      ikdtree.Delete_Point_Boxes(cube_remove);
    }
  }

  void point_body_to_world(PointType const *const pi, PointType *const po) {
    Eigen::Vector3d p_body(pi->x, pi->y, pi->z);
    Eigen::Vector3d p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) +
                             state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
  }

  void point_body_lidar_to_imu(PointType const *const pi, PointType *const po) {
    Eigen::Vector3d p_body_lidar(pi->x, pi->y, pi->z);
    Eigen::Vector3d p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
  }

  void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::UniquePtr msg) {
    mtx_buffer.lock();
    double timestamp = get_time_sec(msg->header.stamp);
    if (timestamp < last_timestamp_lidar) {
      RCLCPP_ERROR(get_logger(), "Lidar timestamp drift detected, clear buffer.");
      lidar_buffer.clear();
    }

    PointCloud::Ptr ptr(new PointCloud());
    p_laser_->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(timestamp);
    last_timestamp_lidar = timestamp;
    mtx_buffer.unlock();

    RCLCPP_DEBUG(get_logger(), "Received lidar stamp @ %f", rclcpp::Time(msg->header.stamp).seconds());
  }

  void imu_cbk(const sensor_msgs::msg::Imu::UniquePtr msg_in) {
    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));
    msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_offset_lidar_to_imu);
    double timestamp = get_time_sec(msg->header.stamp);

    mtx_buffer.lock();
    if (timestamp < last_timestamp_imu) {
      RCLCPP_ERROR(get_logger(), "IMU timestamp drift detected, clear buffer.");
      imu_buffer.clear();
    }
    last_timestamp_imu = timestamp;
    imu_buffer.push_back(msg);
    mtx_buffer.unlock();

    RCLCPP_DEBUG(get_logger(), "Received IMU stamp @ %f", rclcpp::Time(msg_in->header.stamp).seconds());
  }

  void pose_cbk(const geometry_msgs::msg::PoseWithCovarianceStamped::UniquePtr msg_in) {
    static bool pose_inited = false;

    if (pose_inited == false) {
      esekfom::state_ikfom init_state;
      auto position = msg_in->pose.pose.position;
      init_state.pos = Eigen::Vector3d(position.x, position.y, position.z);
      auto orientation = msg_in->pose.pose.orientation;
      Eigen::Quaterniond quat(orientation.w, orientation.x, orientation.y, orientation.z);
      init_state.rot = Sophus::SO3d(quat);
      kf.change_x(init_state);
      pose_inited = true;
      RCLCPP_INFO_STREAM(get_logger(), "Set initial pose:" << init_state.pos.transpose());
    }

    RCLCPP_DEBUG(get_logger(), "Received pose stamp @ %f", rclcpp::Time(msg_in->header.stamp).seconds());
  }

  Eigen::Vector3d so3_to_euler(const Sophus::SO3d &orient) {
    Eigen::Matrix3d R = orient.matrix();
    Eigen::Vector3d euler_rpy = R.eulerAngles(0, 1, 2);
    return euler_rpy;
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_lidar_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_pose_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_frame_map_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_frame_body_;
  // rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_effect_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_pub_;

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::TimerBase::SharedPtr map_pub_timer_;

  std::string lidar_frame_id;
  std::string imu_frame_id;

  bool path_en = true;
  bool scan_pub_en = false;
  bool scan_body_pub_en = false;
  bool dense_pub_en = false;
  // bool effect_pub_en = false;
  bool map_pub_en = false;

  std::shared_ptr<LaserProcess> p_laser_;
  std::shared_ptr<ImuProcess> p_imu_;
  esekfom::esekf kf;
  esekfom::state_ikfom state_point;
  KD_TREE<PointType> ikdtree;

  std::mutex mtx_buffer;
  std::deque<double> time_buffer;
  std::deque<PointCloud::Ptr> lidar_buffer;
  std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;
  MeasureGroup measures;

  PointCloud::Ptr feats_undistort = nullptr;
  PointCloud::Ptr feats_down_body = nullptr;
  PointCloud::Ptr feats_down_world = nullptr;
  PointCloud::Ptr feats_from_map = nullptr;
  PointCloud::Ptr feats_deleted = nullptr;
  PointCloud::Ptr pcl_wait_pub = nullptr;
  nav_msgs::msg::Path path;

  pcl::VoxelGrid<PointType> voxel_filter_surf_;
  pcl::VoxelGrid<PointType> voxel_filter_map_;

  std::ofstream fout_pre;

  int feats_down_size = 0;          // timer_callback->map_incremental
  double first_lidar_time = 0;      // timer_callback->map_incremental
  double lidar_end_time = 0;        // sync_packages->publish
  double last_timestamp_lidar = 0;  // standard_pcl_cbk
  double last_timestamp_imu = -1.0; // imu_cbk->sync_packages

  double cube_len = 0;                 // LaserMappingNode->lasermap_fov_segment
  double DET_RANGE = 300.0f;           // LaserMappingNode->lasermap_fov_segment
  int max_iteration = 0;               // LaserMappingNode->update_iterated_dyn_share_modified
  bool extrinsic_est_en = true;        // LaserMappingNode->update_iterated_dyn_share_modified
  double filter_size_surf_min = 0;     // LaserMappingNode
  double filter_size_map_min = 0;      // LaserMappingNode->map_incremental
  double time_offset_lidar_to_imu = 0; // LaserMappingNode->imu_cbk

  BoxPointType localmap_points;            // lasermap_fov_segment
  bool localmap_inited = false;            // lasermap_fov_segment
  std::vector<PointVector> nearest_points; // update_iterated_dyn_share_modified->map_incremental
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LaserMappingNode>(rclcpp::NodeOptions()));
  return 0;
}
