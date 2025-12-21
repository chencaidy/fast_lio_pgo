#pragma once

#include "common.h"
#include "ikd_tree.h"

#include <Eigen/Core>
#include <sophus/so3.hpp>
#include <vector>

// 该hpp主要包含：广义加减法，前向传播主函数，计算特征点残差及其雅可比，ESKF主函数

namespace esekfom {

typedef Eigen::Matrix<double, 24, 24> cov;             // 24X24的协方差矩阵
typedef Eigen::Matrix<double, 24, 1> vectorized_state; // 24X1的向量

struct state_ikfom {
  Eigen::Vector3d pos = Eigen::Vector3d(0, 0, 0);                        // Position
  Sophus::SO3d rot = Sophus::SO3d(Eigen::Matrix3d::Identity());          // Rotation
  Sophus::SO3d offset_R_L_I = Sophus::SO3d(Eigen::Matrix3d::Identity()); // Extrinsics-R
  Eigen::Vector3d offset_T_L_I = Eigen::Vector3d(0, 0, 0);               // Extrinsics-T
  Eigen::Vector3d vel = Eigen::Vector3d(0, 0, 0);                        // Velocity
  Eigen::Vector3d bg = Eigen::Vector3d(0, 0, 0);                         // Gyro bias
  Eigen::Vector3d ba = Eigen::Vector3d(0, 0, 0);                         // Accel bias
  Eigen::Vector3d grav = Eigen::Vector3d(0, 0, -G_m_s2);                 // Gravity
};

struct input_ikfom {
  Eigen::Vector3d acc = Eigen::Vector3d(0, 0, 0);
  Eigen::Vector3d gyro = Eigen::Vector3d(0, 0, 0);
};

struct dyn_share_datastruct {
  bool valid;                                                // 有效特征点数量是否满足要求
  bool converge;                                             // 迭代时，是否已经收敛
  Eigen::Matrix<double, Eigen::Dynamic, 1> h;                // 残差	(公式(14)中的z)
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> h_x; // 雅可比矩阵H (公式(14)中的H)
};

class esekf {
public:
  esekf();
  ~esekf();

  state_ikfom get_x() { return x_; }

  cov get_P() { return P_; }

  void change_x(state_ikfom &input_state) { x_ = input_state; }

  void change_P(cov &input_cov) { P_ = input_cov; }

  // 前向传播 公式(4-8)
  void predict(double &dt, Eigen::Matrix<double, 12, 12> &Q, const input_ikfom &i_in);

  // ESKF
  void update_iterated_dyn_share_modified(double R, PointCloud::Ptr &feats_down_body, KD_TREE<PointType> &ikdtree,
                                          std::vector<PointVector> &Nearest_Points, int maximum_iter,
                                          bool extrinsic_est);

private:
  // 对应公式(2)的f
  Eigen::Matrix<double, 24, 1> get_f(state_ikfom s, input_ikfom in);

  // 对应公式(7)的Fx  注意该矩阵没乘dt，没加单位阵
  Eigen::Matrix<double, 24, 24> df_dx(state_ikfom s, input_ikfom in);

  // 对应公式(7)的Fw  注意该矩阵没乘dt
  Eigen::Matrix<double, 24, 12> df_dw(state_ikfom s);

  // 计算每个特征点的残差及H矩阵
  void h_share_model(dyn_share_datastruct &ekfom_data, PointCloud::Ptr &feats_down_body, KD_TREE<PointType> &ikdtree,
                     std::vector<PointVector> &Nearest_Points, bool extrinsic_est);

  // 广义加法  公式(4)
  state_ikfom boxplus(state_ikfom x, Eigen::Matrix<double, 24, 1> f_);

  // 广义减法
  vectorized_state boxminus(state_ikfom x1, state_ikfom x2);

  // 拟合平面方程
  template <typename T>
  bool esti_plane(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold);

private:
  state_ikfom x_;
  cov P_ = cov::Identity();

  // 特征点在地图中对应的平面参数（平面的单位法向量，以及当前点到平面距离）
  PointCloud::Ptr normvec = nullptr;
  PointCloud::Ptr laserCloudOri = nullptr; // 有效特征点
  PointCloud::Ptr corr_normvect = nullptr; // 有效特征点对应点法向量
  bool point_selected_surf[100000] = {0};  // 判断是否是有效特征点
};

} // namespace esekfom
