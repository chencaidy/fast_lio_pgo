#include <cmath>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2/LinearMath/Matrix3x3.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2_ros/transform_broadcaster.hpp>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

using std::placeholders::_1;

struct Pose6D {
  double x;
  double y;
  double z;
  double roll;
  double pitch;
  double yaw;
};

typedef pcl::PointXYZI PointType;

class LaserPoseGraphOptimizationNode : public rclcpp::Node {
public:
  LaserPoseGraphOptimizationNode(const rclcpp::NodeOptions &options) : Node("laser_pgo", options) {
    declare_parameter<std::string>("save_directory", "/tmp/laser_pgo");
    declare_parameter<double>("keyframe_meter_gap", 2.0);
    declare_parameter<double>("keyframe_deg_gap", 10.0);
    declare_parameter<double>("mapviz_filter_size", 0.4);

    get_parameter<std::string>("save_directory", save_dir_);
    pg_scans_dir_ = save_dir_ + "/scans";
    int result;
    result = system((std::string("rm -r ") + save_dir_).c_str());
    result = system((std::string("mkdir -p ") + pg_scans_dir_).c_str());
    (void)result;
    pg_kitti_fmt_ = save_dir_ + "/optimized_poses.txt";
    odom_kitti_fmt_ = save_dir_ + "/odom_poses.txt";
    pg_time_stream_ = std::fstream(save_dir_ + "/times.txt", std::fstream::out);
    pg_time_stream_.precision(std::numeric_limits<double>::max_digits10);

    double keyframe_deg_gap;
    get_parameter<double>("keyframe_meter_gap", keyframe_meter_gap_); // Pose assignment every k m move
    get_parameter<double>("keyframe_deg_gap", keyframe_deg_gap);      // Pose assignment every k deg rot
    keyframe_rad_gap_ = keyframe_deg_gap * M_PI / 180.0;

    gtsam::ISAM2Params params;
    params.relinearizeThreshold = 0.01;
    params.relinearizeSkip = 1;
    isam_ = std::make_unique<gtsam::ISAM2>(params);
    init_isam_noises();

    double filter_size = 1.0;
    voxel_filter_input_.setLeafSize(filter_size, filter_size, filter_size);

    double mapviz_filter_size;
    get_parameter<double>("mapviz_filter_size", mapviz_filter_size);
    voxel_filter_mapviz_.setLeafSize(mapviz_filter_size, mapviz_filter_size, mapviz_filter_size);

    sub_laser_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/points", rclcpp::SensorDataQoS(), std::bind(&LaserPoseGraphOptimizationNode::laser_callback, this, _1));
    sub_laser_odom_ = create_subscription<nav_msgs::msg::Odometry>(
        "/odometry", 100, std::bind(&LaserPoseGraphOptimizationNode::laser_odom_callback, this, _1));
    sub_pose_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/pose", 100, std::bind(&LaserPoseGraphOptimizationNode::pose_callback, this, _1));

    pub_odom_ = create_publisher<nav_msgs::msg::Odometry>("/mapping/pgo/odometry", 100);
    pub_path_ = create_publisher<nav_msgs::msg::Path>("/mapping/pgo/path", 100);
    pub_map_ = create_publisher<sensor_msgs::msg::PointCloud2>("/mapping/pgo/points", 100);

    pub_tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    // Pose graph construction
    posegraph_slam = std::make_unique<std::thread>(std::bind(&LaserPoseGraphOptimizationNode::process_pg, this));
    isam_update = std::make_unique<std::thread>(std::bind(&LaserPoseGraphOptimizationNode::process_isam, this));
    // Visualization - map (low frequency because it is heavy)
    viz_map = std::make_unique<std::thread>(std::bind(&LaserPoseGraphOptimizationNode::process_viz_map, this));
    // Visualization - path (high frequency)
    viz_path = std::make_unique<std::thread>(std::bind(&LaserPoseGraphOptimizationNode::process_viz_path, this));
  }

  ~LaserPoseGraphOptimizationNode() {
    posegraph_slam->join();
    isam_update->join();
    viz_map->join();
    viz_path->join();
  }

private:
  std::string pad_zeros(int val, int num_digits = 6) {
    std::ostringstream out;
    out << std::internal << std::setfill('0') << std::setw(num_digits) << val;
    return out.str();
  }

  gtsam::Pose3 pose6d_to_gtsam_pose3(const Pose6D &p) {
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(p.roll, p.pitch, p.yaw), gtsam::Point3(p.x, p.y, p.z));
  }

  void save_odom_vertices_kitti_format(const std::vector<Pose6D> &poses, std::string filename) {
    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(filename.c_str(), std::fstream::out);
    for (const auto &pose6d : poses) {
      gtsam::Pose3 pose = pose6d_to_gtsam_pose3(pose6d);
      gtsam::Point3 t = pose.translation();
      gtsam::Rot3 r = pose.rotation();
      auto col1 = r.column(1); // Point3
      auto col2 = r.column(2); // Point3
      auto col3 = r.column(3); // Point3

      stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " " << col1.y() << " " << col2.y()
             << " " << col3.y() << " " << t.y() << " " << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z()
             << std::endl;
    }
  }

  void save_optimized_vertices_kitti_format(const gtsam::Values &estimates, std::string filename) {
    // ref from gtsam's original code "dataset.cpp"
    std::fstream stream(filename.c_str(), std::fstream::out);

    for (const auto &key_value : estimates) {
      auto p = dynamic_cast<const gtsam::GenericValue<gtsam::Pose3> *>(&key_value.value);
      if (!p)
        continue;

      const gtsam::Pose3 &pose = p->value();

      gtsam::Point3 t = pose.translation();
      gtsam::Rot3 r = pose.rotation();
      auto col1 = r.column(1); // Point3
      auto col2 = r.column(2); // Point3
      auto col3 = r.column(3); // Point3

      stream << col1.x() << " " << col2.x() << " " << col3.x() << " " << t.x() << " " << col1.y() << " " << col2.y()
             << " " << col3.y() << " " << t.y() << " " << col1.z() << " " << col2.z() << " " << col3.z() << " " << t.z()
             << std::endl;
    }
  }

  void save_optimized_vertices_pcdviewer_format(const gtsam::Values &estimates, std::string filename) {
    std::fstream stream(filename.c_str(), std::fstream::out);

    for (const auto &key_value : estimates) {
      auto p = dynamic_cast<const gtsam::GenericValue<gtsam::Pose3> *>(&key_value.value);
      if (!p)
        continue;

      const std::string idx_str = pad_zeros(key_value.key);
      const gtsam::Pose3 &pose = p->value();
      gtsam::Point3 t = pose.translation();
      gtsam::Quaternion r = pose.rotation().toQuaternion();

      stream << idx_str << " " << "0" << " " << t.x() << " " << t.y() << " " << t.z() << " " << r.x() << " " << r.y()
             << " " << r.z() << " " << r.w() << std::endl;
    }
  }

  void laser_odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(buffer_mtx_);
    buffer_odom_.push_back(msg);
    if (buffer_odom_.size() > 100) {
      buffer_odom_.pop_front();
    }
  }

  void laser_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(buffer_mtx_);
    buffer_laser_.push_back(msg);
  }

  void pose_callback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(buffer_mtx_);
    buffer_pose_.push_back(msg);
    if (buffer_pose_.size() > 25) {
      buffer_pose_.pop_front();
    }
  }

  void init_isam_noises(void) {
    gtsam::Vector prior_noise_vector6(6);
    prior_noise_vector6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
    prior_noise_ = gtsam::noiseModel::Diagonal::Variances(prior_noise_vector6);

    gtsam::Vector odom_noise_vector6(6);
    // odom_noise_Vector6 << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
    odom_noise_vector6 << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4;
    odom_noise_ = gtsam::noiseModel::Diagonal::Variances(odom_noise_vector6);

    // GPS factor only has 3 elements (x y z)
    gtsam::Vector robust_noise_vector3(3);
    robust_noise_vector3 << 250.0, 250.0, 250.0;
    robust_gps_noise_ = gtsam::noiseModel::Robust::Create(gtsam::noiseModel::mEstimator::Cauchy::Create(1),
                                                          gtsam::noiseModel::Diagonal::Variances(robust_noise_vector3));
    // Optional: replacing Cauchy by DCS or GemanMcClure is okay but Cauchy is empirically good.
  }

  Pose6D get_odom_to_pose6d(nav_msgs::msg::Odometry::ConstSharedPtr _odom) {
    auto tx = _odom->pose.pose.position.x;
    auto ty = _odom->pose.pose.position.y;
    auto tz = _odom->pose.pose.position.z;

    double roll, pitch, yaw;
    auto quat = _odom->pose.pose.orientation;
    tf2::Matrix3x3(tf2::Quaternion(quat.x, quat.y, quat.z, quat.w)).getRPY(roll, pitch, yaw);

    return Pose6D{tx, ty, tz, roll, pitch, yaw};
  }

  Pose6D get_transform_diff(const Pose6D &_p1, const Pose6D &_p2) {
    Eigen::Affine3f SE3_p1 = pcl::getTransformation(_p1.x, _p1.y, _p1.z, _p1.roll, _p1.pitch, _p1.yaw);
    Eigen::Affine3f SE3_p2 = pcl::getTransformation(_p2.x, _p2.y, _p2.z, _p2.roll, _p2.pitch, _p2.yaw);
    Eigen::Matrix4f SE3_delta0 = SE3_p1.matrix().inverse() * SE3_p2.matrix();
    Eigen::Affine3f SE3_delta;
    SE3_delta.matrix() = SE3_delta0;
    float dx, dy, dz, droll, dpitch, dyaw;
    pcl::getTranslationAndEulerAngles(SE3_delta, dx, dy, dz, droll, dpitch, dyaw);
    // std::cout << "delta : " << dx << ", " << dy << ", " << dz << ", " << droll << ", " << dpitch << ", " << dyaw <<
    // std::endl;

    return Pose6D{double(abs(dx)),    double(abs(dy)),     double(abs(dz)),
                  double(abs(droll)), double(abs(dpitch)), double(abs(dyaw))};
  }

  pcl::PointCloud<PointType>::Ptr pcl_local_to_global(const pcl::PointCloud<PointType>::Ptr &cloudIn,
                                                      const Pose6D &tf) {
    pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());

    int cloudSize = cloudIn->size();
    cloudOut->resize(cloudSize);

    Eigen::Affine3f transCur = pcl::getTransformation(tf.x, tf.y, tf.z, tf.roll, tf.pitch, tf.yaw);

#pragma omp parallel for num_threads(16)
    for (int i = 0; i < cloudSize; ++i) {
      const auto &pointFrom = cloudIn->points[i];
      cloudOut->points[i].x =
          transCur(0, 0) * pointFrom.x + transCur(0, 1) * pointFrom.y + transCur(0, 2) * pointFrom.z + transCur(0, 3);
      cloudOut->points[i].y =
          transCur(1, 0) * pointFrom.x + transCur(1, 1) * pointFrom.y + transCur(1, 2) * pointFrom.z + transCur(1, 3);
      cloudOut->points[i].z =
          transCur(2, 0) * pointFrom.x + transCur(2, 1) * pointFrom.y + transCur(2, 2) * pointFrom.z + transCur(2, 3);
      cloudOut->points[i].intensity = pointFrom.intensity;
    }

    return cloudOut;
  }

  void process_pg() {
    rclcpp::Rate rate(100);
    while (rclcpp::ok()) {
      rate.sleep();
      while (1) {
        sensor_msgs::msg::PointCloud2::ConstSharedPtr current_laser = nullptr;
        nav_msgs::msg::Odometry::ConstSharedPtr current_odom = nullptr;
        geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr current_pose = nullptr;

        // Time sync process
        {
          std::lock_guard<std::mutex> lock(buffer_mtx_);

          // Exception handle
          if (buffer_laser_.empty() || buffer_odom_.empty() || buffer_pose_.empty()) {
            break;
          }

          current_laser = buffer_laser_.front();
          double laser_stamp = rclcpp::Time(current_laser->header.stamp).seconds();

          double odom_stamp_first = rclcpp::Time(buffer_odom_.front()->header.stamp).seconds();
          if (laser_stamp < odom_stamp_first) {
            RCLCPP_ERROR(get_logger(), "Lidar out of sync dropped @ %f", laser_stamp);
            buffer_laser_.pop_front();
            break;
          }

          double odom_stamp_last = rclcpp::Time(buffer_odom_.back()->header.stamp).seconds();
          if (laser_stamp > odom_stamp_last) {
            break;
          }

          double pose_stamp_last = rclcpp::Time(buffer_pose_.back()->header.stamp).seconds();
          if (laser_stamp > pose_stamp_last) {
            break;
          }

          // Find closest odometry frame
          for (const auto &odom : buffer_odom_) {
            double eps = 0.01;
            double odom_stamp = rclcpp::Time(odom->header.stamp).seconds();
            if (abs(laser_stamp - odom_stamp) < eps) {
              current_odom = odom;
            }
          }

          // Find closest global pose frame
          for (const auto &pose : buffer_pose_) {
            double eps = 0.02;
            double pose_stamp = rclcpp::Time(pose->header.stamp).seconds() - 0.075;
            if (abs(laser_stamp - pose_stamp) < eps) {
              current_pose = pose;
            }
          }

          // Availability check
          if (current_odom == nullptr) {
            RCLCPP_ERROR(get_logger(), "Failed to get odometry frame @ %f", laser_stamp);
            buffer_laser_.pop_front();
            break;
          }

          buffer_laser_.pop_front();
        }

        double laser_stamp = rclcpp::Time(current_laser->header.stamp).seconds();
        double odom_stamp = rclcpp::Time(current_odom->header.stamp).seconds();
        double pose_stamp = current_pose ? rclcpp::Time(current_pose->header.stamp).seconds() : 0.0;
        RCLCPP_DEBUG(get_logger(), "Lidar_TS=%f Odom_TS=%f Pose_TS=%f", laser_stamp, odom_stamp, pose_stamp);

        // Early reject by counting local delta movement (for equi-spereated kf drop)
        Pose6D this_odom = get_odom_to_pose6d(current_odom);
        Pose6D dtf = get_transform_diff(prev_odom_, this_odom);                         // dtf means delta_transform
        double delta_translation = sqrt(dtf.x * dtf.x + dtf.y * dtf.y + dtf.z * dtf.z); // note: absolute value.
        translation_accumulated_ += delta_translation;
        rotaion_accumulated_ += (dtf.roll + dtf.pitch + dtf.yaw); // sum just naive approach.
        prev_odom_ = this_odom;

        bool is_keyframe = false;
        if (translation_accumulated_ > keyframe_meter_gap_ || rotaion_accumulated_ > keyframe_rad_gap_ ||
            current_pose) {
          is_keyframe = true;
          translation_accumulated_ = 0.0; // reset
          rotaion_accumulated_ = 0.0;     // reset
        }
        if (!is_keyframe)
          continue;

        // Points convert
        pcl::PointCloud<PointType>::Ptr this_keyframe(new pcl::PointCloud<PointType>());
        pcl::fromROSMsg(*current_laser, *this_keyframe);

        // Points downsample
        pcl::PointCloud<PointType>::Ptr this_keyframe_ds(new pcl::PointCloud<PointType>());
        // voxel_filter_input_.setInputCloud(this_keyframe);
        // voxel_filter_input_.filter(*this_keyframe_ds);
        pcl::PassThrough<PointType> pass;
        pass.setInputCloud(this_keyframe);
        pass.setFilterFieldName("intensity");
        pass.setFilterLimits(0.0, 20.0);
        pass.setNegative(true);
        pass.filter(*this_keyframe_ds);

        keyframe_mtx_.lock();
        keyframes_laser_.push_back(this_keyframe_ds);
        keyframes_pose_.push_back(this_odom);
        keyframes_pose_updated_.push_back(this_odom); // Add for initial and update at another thread
        keyframes_time_.push_back(odom_stamp);
        keyframe_mtx_.unlock();

        const int prev_node_idx = keyframes_pose_.size() - 2;
        const int curr_node_idx = keyframes_pose_.size() - 1;
        if (!isam_initialized_) {
          /* Prior node */
          const int init_node_idx = 0;
          gtsam::Pose3 pose_origin = pose6d_to_gtsam_pose3(keyframes_pose_.at(init_node_idx));

          {
            std::lock_guard<std::mutex> lock(isam_mtx_);
            // Prior factor
            isam_graph_.add(gtsam::PriorFactor<gtsam::Pose3>(init_node_idx, pose_origin, prior_noise_));
            isam_initial_estimate_.insert(init_node_idx, pose_origin);
          }

          isam_initialized_ = true;
          RCLCPP_INFO(get_logger(), "Posegraph prior node %d added", init_node_idx);
        } else {
          /* Consecutive node (and odom factor) after the prior added */
          gtsam::Pose3 pose_from = pose6d_to_gtsam_pose3(keyframes_pose_.at(prev_node_idx));
          gtsam::Pose3 pose_to = pose6d_to_gtsam_pose3(keyframes_pose_.at(curr_node_idx));

          {
            std::lock_guard<std::mutex> lock(isam_mtx_);
            // Odom factor
            isam_graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, pose_from.between(pose_to),
                                                               odom_noise_));
            isam_initial_estimate_.insert(curr_node_idx, pose_to);
            // Global pose factor
            if (current_pose != nullptr) {
              auto conv = current_pose->pose.covariance;
              if (conv[0] < 0.5 && conv[7] < 0.5 && conv[14] < 0.5) {
                auto position = current_pose->pose.pose.position;
                gtsam::Point3 pose(position.x, position.y, position.z);
                isam_graph_.add(gtsam::GPSFactor(curr_node_idx, pose, robust_gps_noise_));
                RCLCPP_INFO(get_logger(), "GPS factor added at node %d X=%f Y=%f Z=%f", curr_node_idx, pose.x(),
                            pose.y(), pose.z());
              }
            }
          }

          RCLCPP_INFO(get_logger(), "Posegraph odom node %d added", curr_node_idx);
        }
        // If want to print the current graph, use isam_graph_.print("\nFactor Graph:\n");

        // Save utility
        std::string curr_node_idx_str = pad_zeros(curr_node_idx);
        pcl::io::savePCDFileBinary(pg_scans_dir_ + "/" + curr_node_idx_str + ".pcd", *this_keyframe);
        pg_time_stream_ << laser_stamp << std::endl;
      }
    }
  }

#if 0
  void process_icp(void) {
    while (rclcpp::ok()) {
      while (!scLoopICPBuf.empty()) {
        if (scLoopICPBuf.size() > 30) {
          RCLCPP_WARN(get_logger(), "Too many loop clousre candidates. Do process_lcd less frequently");
        }

        buffer_mtx_.lock();
        std::pair<int, int> loop_idx_pair = scLoopICPBuf.front();
        scLoopICPBuf.pop();
        buffer_mtx_.unlock();

        const int prev_node_idx = loop_idx_pair.first;
        const int curr_node_idx = loop_idx_pair.second;
        auto relative_pose_optional = doICPVirtualRelative(prev_node_idx, curr_node_idx);
        if (relative_pose_optional) {
          gtsam::Pose3 relative_pose = relative_pose_optional.value();
          isam_mtx_.lock();
          isam_graph_.add(
              gtsam::BetweenFactor<gtsam::Pose3>(prev_node_idx, curr_node_idx, relative_pose, robustLoopNoise));
          // run_isam2_optimize();
          isam_mtx_.unlock();
        }
      }

      // wait (must required for running the while loop)
      std::chrono::milliseconds dura(2);
      std::this_thread::sleep_for(dura);
    }
  }
#endif

  void update_keyframes_pose(const gtsam::Values &estimate) {
    std::lock_guard<std::mutex> lock(keyframe_mtx_);

    for (int node_idx = 0; node_idx < int(estimate.size()); node_idx++) {
      Pose6D &p = keyframes_pose_updated_[node_idx];
      p.x = estimate.at<gtsam::Pose3>(node_idx).translation().x();
      p.y = estimate.at<gtsam::Pose3>(node_idx).translation().y();
      p.z = estimate.at<gtsam::Pose3>(node_idx).translation().z();
      p.roll = estimate.at<gtsam::Pose3>(node_idx).rotation().roll();
      p.pitch = estimate.at<gtsam::Pose3>(node_idx).rotation().pitch();
      p.yaw = estimate.at<gtsam::Pose3>(node_idx).rotation().yaw();
    }

    recent_idx_updated_ = int(keyframes_pose_updated_.size()) - 1;
  }

  void process_isam(void) {
    rclcpp::Rate rate(1);
    while (rclcpp::ok()) {
      rate.sleep();
      if (isam_initialized_) {
        RCLCPP_INFO(get_logger(), "Running isam2 optimization...");
        {
          std::lock_guard<std::mutex> lock(isam_mtx_);
          isam_->update(isam_graph_, isam_initial_estimate_);
          isam_graph_.resize(0);
          isam_initial_estimate_.clear();
        }

        // Save optimized result
        auto isam_current_estimate = isam_->calculateEstimate();
        update_keyframes_pose(isam_current_estimate);
        save_optimized_vertices_pcdviewer_format(isam_current_estimate, pg_kitti_fmt_);

        // Save origin result
        {
          std::lock_guard<std::mutex> lock(keyframe_mtx_);
          save_odom_vertices_kitti_format(keyframes_pose_, odom_kitti_fmt_);
        }
      }
    }
  }

  void publish_path(void) {
    nav_msgs::msg::Odometry pgo_odom;
    nav_msgs::msg::Path pgo_path;

    keyframe_mtx_.lock();
    for (int node_idx = 0; node_idx < recent_idx_updated_; node_idx++) {
      const Pose6D &pose_est = keyframes_pose_updated_.at(node_idx); // Updated poses

      nav_msgs::msg::Odometry odom;
      odom.header.frame_id = "map";
      odom.child_frame_id = "aft_pgo";
      odom.header.stamp = rclcpp::Time(keyframes_time_.at(node_idx) * 1e9);
      odom.pose.pose.position.x = pose_est.x;
      odom.pose.pose.position.y = pose_est.y;
      odom.pose.pose.position.z = pose_est.z;
      tf2::Quaternion quat;
      quat.setRPY(pose_est.roll, pose_est.pitch, pose_est.yaw);
      quat.normalize();
      odom.pose.pose.orientation.w = quat.getW();
      odom.pose.pose.orientation.x = quat.getX();
      odom.pose.pose.orientation.y = quat.getY();
      odom.pose.pose.orientation.z = quat.getZ();
      pgo_odom = odom;

      geometry_msgs::msg::PoseStamped pose;
      pose.header = odom.header;
      pose.pose = odom.pose.pose;

      pgo_path.header.stamp = odom.header.stamp;
      pgo_path.header.frame_id = "map";
      pgo_path.poses.push_back(pose);
    }
    keyframe_mtx_.unlock();

    pub_odom_->publish(pgo_odom);
    pub_path_->publish(pgo_path);

    geometry_msgs::msg::TransformStamped trans;
    trans.header.frame_id = "map";
    trans.child_frame_id = "aft_pgo";
    trans.header.stamp = pgo_odom.header.stamp;
    trans.transform.translation.x = pgo_odom.pose.pose.position.x;
    trans.transform.translation.y = pgo_odom.pose.pose.position.y;
    trans.transform.translation.z = pgo_odom.pose.pose.position.z;
    trans.transform.rotation = pgo_odom.pose.pose.orientation;
    pub_tf_->sendTransform(trans);
  }

  void process_viz_path(void) {
    rclcpp::Rate rate(10);
    while (rclcpp::ok()) {
      rate.sleep();
      if (recent_idx_updated_ > 1) {
        publish_path();
      }
    }
  }

  void publish_map(void) {
    static int last_node_idx = 0;
    static pcl::PointCloud<PointType> pgo_points;
    int SKIP_FRAMES = 10; // sparse map visulalization to save computations
    int counter = 0;

    keyframe_mtx_.lock();
    for (int node_idx = last_node_idx; node_idx < recent_idx_updated_; node_idx++) {
      if (counter % SKIP_FRAMES == 0) {
        pcl::PointCloud<PointType>::Ptr points(new pcl::PointCloud<PointType>());
        voxel_filter_mapviz_.setInputCloud(keyframes_laser_[node_idx]);
        voxel_filter_mapviz_.filter(*points);
        pgo_points += *pcl_local_to_global(points, keyframes_pose_updated_[node_idx]);
      }
      counter++;
    }
    last_node_idx = recent_idx_updated_ - 1;
    keyframe_mtx_.unlock();

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(pgo_points, msg);
    msg.header.frame_id = "map";
    pub_map_->publish(msg);
  }

  void process_viz_map(void) {
    rclcpp::Rate rate(0.2);
    while (rclcpp::ok()) {
      rate.sleep();
      if (recent_idx_updated_ > 1) {
        publish_map();
      }
    }
  }

private:
  double keyframe_meter_gap_, keyframe_rad_gap_;
  double translation_accumulated_ = 1000000.0;     // Large value means must add the first given frame.
  double rotaion_accumulated_ = 1000000.0;         // Large value means must add the first given frame.
  Pose6D prev_odom_{0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // Init

  std::mutex buffer_mtx_;
  std::deque<nav_msgs::msg::Odometry::ConstSharedPtr> buffer_odom_;
  std::deque<sensor_msgs::msg::PointCloud2::ConstSharedPtr> buffer_laser_;
  std::deque<geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr> buffer_pose_;

  std::mutex keyframe_mtx_;
  std::vector<pcl::PointCloud<PointType>::Ptr> keyframes_laser_;
  std::vector<Pose6D> keyframes_pose_;
  std::vector<Pose6D> keyframes_pose_updated_;
  std::vector<double> keyframes_time_;
  int recent_idx_updated_ = 0;

  std::mutex isam_mtx_;
  std::unique_ptr<gtsam::ISAM2> isam_;
  gtsam::NonlinearFactorGraph isam_graph_;
  gtsam::Values isam_initial_estimate_;
  bool isam_initialized_ = false;

  gtsam::noiseModel::Diagonal::shared_ptr prior_noise_;
  gtsam::noiseModel::Diagonal::shared_ptr odom_noise_;
  gtsam::noiseModel::Base::shared_ptr robust_gps_noise_;

  pcl::VoxelGrid<PointType> voxel_filter_input_;
  pcl::VoxelGrid<PointType> voxel_filter_mapviz_;

  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pub_path_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_laser_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_laser_odom_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr sub_pose_;

  std::unique_ptr<tf2_ros::TransformBroadcaster> pub_tf_;

  std::unique_ptr<std::thread> posegraph_slam;
  std::unique_ptr<std::thread> isam_update;
  std::unique_ptr<std::thread> viz_map;
  std::unique_ptr<std::thread> viz_path;

  std::string save_dir_;
  std::string pg_kitti_fmt_, pg_scans_dir_;
  std::string odom_kitti_fmt_;
  std::fstream pg_time_stream_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LaserPoseGraphOptimizationNode>(rclcpp::NodeOptions()));
  return 0;
}
