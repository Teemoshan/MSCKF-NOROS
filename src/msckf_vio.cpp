/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <iterator>
#include <algorithm>
#include <fstream>

#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/SparseCore>
//#include <Eigen/SPQRSupport>
#include "boost/math/distributions/chi_squared.hpp"

#include <msckf_vio/msckf_vio.h>
#include <msckf_vio/math_utils.hpp>
#include <msckf_vio/utils.h>


using namespace Eigen;

namespace msckf_vio{
// Static member variables in IMUState class.
StateIDType IMUState::next_id = 0;
double IMUState::gyro_noise = 0;
double IMUState::acc_noise = 0;
double IMUState::gyro_bias_noise = 0;
double IMUState::acc_bias_noise = 0;
Vector3d IMUState::gravity = Vector3d(0, 0, -GRAVITY_ACCELERATION);
Isometry3d IMUState::T_imu_body = Isometry3d::Identity();
// Static member variables in CAMState class.
Isometry3d CAMState::T_cam0_cam1 = Isometry3d::Identity();
// Static member variables in Feature class.
FeatureIDType Feature::next_id = 0;
double Feature::observation_noise = 0.01;
Feature::OptimizationConfig Feature::optimization_config;
map<int, double> MsckfVio::chi_squared_test_table;
//DEBUG HAN
string MSCKF_RESULT_PATH="./msckfvio.csv";


MsckfVio::MsckfVio():
  is_gravity_set(false),
  is_first_img(true)
  {
  initialize();
  return;
}

bool MsckfVio::loadParameters() {
  // Frame id
  //
  position_std_threshold=8.0;
  rotation_threshold=0.2618;
  translation_threshold=0.4;
  tracking_rate_threshold=0.5;

  // Feature optimization parameters  
  Feature::optimization_config.translation_threshold=-0.1;

  // Noise related parameters
  //改噪声
  IMUState::gyro_noise=0.005;
  IMUState::acc_noise=0.05;
  IMUState::gyro_bias_noise= 0.001;
  IMUState::acc_bias_noise= 0.01;
  Feature::observation_noise=0.035;

  // Use variance instead of standard deviation.
  IMUState::gyro_noise *= IMUState::gyro_noise;
  IMUState::acc_noise *= IMUState::acc_noise;
  IMUState::gyro_bias_noise *= IMUState::gyro_bias_noise;
  IMUState::acc_bias_noise *= IMUState::acc_bias_noise;
  Feature::observation_noise *= Feature::observation_noise;

  // Set the initial IMU state.
  // The intial orientation and position will be set to the origin
  // implicitly. But the initial velocity and bias can be
  // set by parameters.
  // TODO: is it reasonable to set the initial bias to 0?

      state_server.imu_state.velocity(0)= 0.0;

      state_server.imu_state.velocity(1)= 0.0;

      state_server.imu_state.velocity(2)= 0.0;

  // The initial covariance of orientation and position can be
  // set to 0. But for velocity, bias and extrinsic parameters,
  // there should be nontrivial uncertainty.
  double gyro_bias_cov, acc_bias_cov, velocity_cov;  
      velocity_cov=0.25;
      gyro_bias_cov=0.01;
      acc_bias_cov= 0.01;

  double extrinsic_rotation_cov, extrinsic_translation_cov;  
      extrinsic_rotation_cov=0.000305;
      extrinsic_translation_cov=0.000025;

  state_server.state_cov = MatrixXd::Zero(21, 21);

  for (int i = 3; i < 6; ++i)
    state_server.state_cov(i, i) = gyro_bias_cov;
  for (int i = 6; i < 9; ++i)
    state_server.state_cov(i, i) = velocity_cov;
  for (int i = 9; i < 12; ++i)
    state_server.state_cov(i, i) = acc_bias_cov;
  for (int i = 15; i < 18; ++i)
    state_server.state_cov(i, i) = extrinsic_rotation_cov;
  for (int i = 18; i < 21; ++i)
    state_server.state_cov(i, i) = extrinsic_translation_cov;

  // Transformation offsets between the frames involved.
  //改从imu到相机0的变换矩阵
  Eigen::Matrix3d R3;
  R3 <<  0.014865542981794,   0.999557249008346,  -0.025774436697440,
       -0.999880929698575,   0.014967213324719,   0.003756188357967,
       0.004140296794224,   0.025715529947966,   0.999660727177902;
  Eigen::Vector3d T3;
  T3 << 0.065222909535531,  -0.020706385492719,  -0.008054602460030;
  Isometry3d T_imu_cam0=Isometry3d::Identity();
  T_imu_cam0.prerotate(R3);
  T_imu_cam0.pretranslate(T3);
  Isometry3d T_cam0_imu = T_imu_cam0.inverse();


  state_server.imu_state.R_imu_cam0 = T_cam0_imu.linear().transpose();
  state_server.imu_state.t_cam0_imu = T_cam0_imu.translation();
  //改从相机0到相机1的变换矩阵
  Eigen::Matrix4d T4;
  T4 << 0.999997256477881,   0.002312067192424,   0.000376008102415,  -0.110073808127187,
          -0.002317135723281,   0.999898048506644,   0.014089835846648,   0.000399121547014,
          -0.000343393120525,  -0.014090668452714,   0.999900662637729,  -0.000853702503357,
                           0,                   0,                   0,   1.000000000000000;
  CAMState::T_cam0_cam1 =T4;

  Eigen::Matrix4d T5;
  T5 <<   1.0000, 0.0000, 0.0000, 0.0000,
          0.0000, 1.0000, 0.0000, 0.0000,
          0.0000, 0.0000, 1.0000, 0.0000,
          0.0000, 0.0000, 0.0000, 1.0000;
  IMUState::T_imu_body =
    T5.inverse();

  // Maximum number of camera states to be stored
   max_cam_state_size=20;

  std::cout<<"R_imu_cam0"<<state_server.imu_state.R_imu_cam0<<std::endl;
  std::cout<<"t_cam0_imu"<<state_server.imu_state.t_cam0_imu[0]<<state_server.imu_state.t_cam0_imu[1]<<
  state_server.imu_state.t_cam0_imu[2]<<std::endl;
 // std::cout<<"T_cam0_cam1"<<CAMState::T_cam0_cam1<<std::endl;

  return true;
}


bool MsckfVio::initialize() {
  if (!loadParameters()) return false;
  std::cout<<"Finish loading  parameters..."<<std::endl;

  // Initialize state server
  state_server.continuous_noise_cov =
    Matrix<double, 12, 12>::Zero();
  state_server.continuous_noise_cov.block<3, 3>(0, 0) =
    Matrix3d::Identity()*IMUState::gyro_noise;
  state_server.continuous_noise_cov.block<3, 3>(3, 3) =
    Matrix3d::Identity()*IMUState::gyro_bias_noise;
  state_server.continuous_noise_cov.block<3, 3>(6, 6) =
    Matrix3d::Identity()*IMUState::acc_noise;
  state_server.continuous_noise_cov.block<3, 3>(9, 9) =
    Matrix3d::Identity()*IMUState::acc_bias_noise;

  // Initialize the chi squared test table with confidence
  // level 0.95.
  for (int i = 1; i < 100; ++i) {
    boost::math::chi_squared chi_squared_dist(i);
    chi_squared_test_table[i] =
      boost::math::quantile(chi_squared_dist, 0.05);
  } 
  return true;
}

void MsckfVio::imuCallback(
    const double timestamp, const Eigen::Vector3d& accl, const Eigen::Vector3d& gyro) {

  // IMU msgs are pushed backed into a buffer instead of
  // being processed immediately. The IMU msgs are processed
  // when the next image is available, in which way, we can
  // easily handle the transfer delay.
   // std::cout<<"imutime="<<timestamp<<std::endl;

    Eigen::Matrix<double, 6, 1> tempimu;
    tempimu<<accl[0],accl[1],accl[2],gyro[0],gyro[1],gyro[2];
    imubuffer.push_back(make_pair(timestamp,tempimu));

  if (!is_gravity_set) {
    if (imubuffer.size() < 200) return;
    //if (imu_msg_buffer.size() < 10) return;
    std::cout<<"initial g"<<std::endl;
    initializeGravityAndBias();
    is_gravity_set = true;
  }

  return;
}
//静止时候初始化重力方向，陀螺仪偏值，从重力坐标系到IMU坐标系
void MsckfVio::initializeGravityAndBias() {

  // Initialize gravity and gyro bias.
  Vector3d sum_angular_vel = Vector3d::Zero();
  Vector3d sum_linear_acc = Vector3d::Zero();

  for (const auto& imu_msg : imubuffer) {
    Vector3d angular_vel = Vector3d::Zero();
    Vector3d linear_acc = Vector3d::Zero();
    linear_acc << imu_msg.second[0],imu_msg.second[1],imu_msg.second[2];
    angular_vel << imu_msg.second[3],imu_msg.second[4],imu_msg.second[5];
    sum_angular_vel += angular_vel;
    sum_linear_acc += linear_acc;
  }

  state_server.imu_state.gyro_bias =
    sum_angular_vel / imubuffer.size();
  //IMUState::gravity =
  //  -sum_linear_acc / imu_msg_buffer.size();
  // This is the gravity in the IMU frame.
  Vector3d gravity_imu =
    sum_linear_acc / imubuffer.size();

  // Initialize the initial orientation, so that the estimation
  // is consistent with the inertial frame.
  double gravity_norm = gravity_imu.norm();
  IMUState::gravity = Vector3d(0.0, 0.0, -gravity_norm);
  //FromTwoVectors()是从第一个向量转到第二个向量
  //即是从imu坐标系到重力坐标系
  Quaterniond q0_i_w = Quaterniond::FromTwoVectors(
    gravity_imu, -IMUState::gravity);
  state_server.imu_state.orientation =
    rotationToQuaternion(q0_i_w.toRotationMatrix().transpose());

  return;
}


void MsckfVio::featureCallback(
    const vector<pair<double, std::vector<Eigen::Matrix<double, 5, 1>>> > & msg) {

  // Return if the gravity vector has not been set.
  if (!is_gravity_set) return;

  cout << "==================================" << endl;
  cout << "start new state estimate" << endl;
  cout << "==================================" << endl;
  // Start the system if the first image is received.
  // The frame where the first image is received will be
  // the origin.
  if (is_first_img) {
    is_first_img = false;
    state_server.imu_state.time = msg[0].first;
  }



  TicToc t_whole ; 
  // Propogate the IMU state.
  // that are received before the image msg.

  batchImuProcessing(msg[0].first);


  // Augment the state vector.

  stateAugmentation(msg[0].first);


  // Add new observations for existing features or new
  // features in the map server.

  addFeatureObservations(msg);


  // Perform measurement update if necessary.

  removeLostFeatures();



  pruneCamStateBuffer();

  // Publish the odometry.

  publish(msg[0].first);

  std::cout<<"whole time for state estimate"<<t_whole.toc()<<std::endl;
  // Reset the system if necessary.
 
  return;
}


//得到图像帧时间间隔内的imu数据
void MsckfVio::batchImuProcessing(const double& time_bound) {
  // Counter how many IMU msgs in the buffer are used.
  int used_imu_msg_cntr = 0;
  //std::cout<<imubuffer.size()<<std::endl;
  for (const auto& imu_msg : imubuffer) {
    double imu_time = imu_msg.first;
    if (imu_time < state_server.imu_state.time) {
      ++used_imu_msg_cntr;
      continue;
    }
    if (imu_time > time_bound) break;

    // Convert the msgs.
    Vector3d m_gyro, m_acc;
    m_gyro<<imu_msg.second[3],imu_msg.second[4],imu_msg.second[5];
    m_acc<<imu_msg.second[0],imu_msg.second[1],imu_msg.second[2];


    // Execute process model.
    processModel(imu_time, m_gyro, m_acc);

    ++used_imu_msg_cntr;
  }
  //std::cout<<"propoimu"<<used_imu_msg_cntr<<std::endl;
  // Set the state ID for the new IMU state.
  state_server.imu_state.id = IMUState::next_id++;

  // Remove all used IMU msgs.
  imubuffer.erase(imubuffer.begin(),
      imubuffer.begin()+used_imu_msg_cntr);
     

  return;
}

void MsckfVio::processModel(const double& time,
    const Vector3d& m_gyro,
    const Vector3d& m_acc) {

  // Remove the bias from the measured gyro and acceleration
  //1.将加速度和陀螺仪减去他们的偏值
  IMUState& imu_state = state_server.imu_state;
  Vector3d gyro = m_gyro - imu_state.gyro_bias;
  Vector3d acc = m_acc - imu_state.acc_bias;
  double dtime = time - imu_state.time;

  // Compute discrete transition and noise covariance matrix
  //2.计算IMU状态量的倒数的雅可比和协方差
  Matrix<double, 21, 21> F = Matrix<double, 21, 21>::Zero();
  Matrix<double, 21, 12> G = Matrix<double, 21, 12>::Zero();

  F.block<3, 3>(0, 0) = -skewSymmetric(gyro);
  F.block<3, 3>(0, 3) = -Matrix3d::Identity();
  F.block<3, 3>(6, 0) = -quaternionToRotation(
      imu_state.orientation).transpose()*skewSymmetric(acc);
  F.block<3, 3>(6, 9) = -quaternionToRotation(
      imu_state.orientation).transpose();
  F.block<3, 3>(12, 6) = Matrix3d::Identity();

  G.block<3, 3>(0, 0) = -Matrix3d::Identity();
  G.block<3, 3>(3, 3) = Matrix3d::Identity();
  G.block<3, 3>(6, 6) = -quaternionToRotation(
      imu_state.orientation).transpose();
  G.block<3, 3>(9, 9) = Matrix3d::Identity();

  // Approximate matrix exponential to the 3rd order,
  // which can be considered to be accurate enough assuming
  // dtime is within 0.01s.
  //3.状态方程的离散化（3阶，vins是1阶）
  Matrix<double, 21, 21> Fdt = F * dtime;
  Matrix<double, 21, 21> Fdt_square = Fdt * Fdt;
  Matrix<double, 21, 21> Fdt_cube = Fdt_square * Fdt;
  Matrix<double, 21, 21> Phi = Matrix<double, 21, 21>::Identity() +
    Fdt + 0.5*Fdt_square + (1.0/6.0)*Fdt_cube;

  // Propogate the state using 4th order Runge-Kutta
  //4.RK4积分p,v,q
  predictNewState(dtime, gyro, acc);

  // Modify the transition matrix
  //5.改进状态转移矩阵Phi
  //？？？
  Matrix3d R_kk_1 = quaternionToRotation(imu_state.orientation_null);
  Phi.block<3, 3>(0, 0) =
    quaternionToRotation(imu_state.orientation) * R_kk_1.transpose();

  Vector3d u = R_kk_1 * IMUState::gravity;
  RowVector3d s = (u.transpose()*u).inverse() * u.transpose();

  Matrix3d A1 = Phi.block<3, 3>(6, 0);
  Vector3d w1 = skewSymmetric(
      imu_state.velocity_null-imu_state.velocity) * IMUState::gravity;
  Phi.block<3, 3>(6, 0) = A1 - (A1*u-w1)*s;

  Matrix3d A2 = Phi.block<3, 3>(12, 0);
  Vector3d w2 = skewSymmetric(
      dtime*imu_state.velocity_null+imu_state.position_null-
      imu_state.position) * IMUState::gravity;
  Phi.block<3, 3>(12, 0) = A2 - (A2*u-w2)*s;

  // Propogate the state covariance matrix.
  //6.协方差更新
  //IMU状态协方差更新
  Matrix<double, 21, 21> Q = Phi*G*state_server.continuous_noise_cov*
    G.transpose()*Phi.transpose()*dtime;
  state_server.state_cov.block<21, 21>(0, 0) =
    Phi*state_server.state_cov.block<21, 21>(0, 0)*Phi.transpose() + Q;
  //相机状态协方差更新
  if (state_server.cam_states.size() > 0) {
    state_server.state_cov.block(
        0, 21, 21, state_server.state_cov.cols()-21) =
      Phi * state_server.state_cov.block(
        0, 21, 21, state_server.state_cov.cols()-21);
    state_server.state_cov.block(
        21, 0, state_server.state_cov.rows()-21, 21) =
      state_server.state_cov.block(
        21, 0, state_server.state_cov.rows()-21, 21) * Phi.transpose();
  }
  //7.保证协方差的对称性
  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;



  // Update the state correspondes to null space.
  imu_state.orientation_null = imu_state.orientation;
  imu_state.position_null = imu_state.position;
  imu_state.velocity_null = imu_state.velocity;

  // Update the state info
  state_server.imu_state.time = time;
  return;
}

void MsckfVio::predictNewState(const double& dt,
    const Vector3d& gyro,
    const Vector3d& acc) {

  // TODO: Will performing the forward integration using
  //    the inverse of the quaternion give better accuracy?
  double gyro_norm = gyro.norm();
  Matrix4d Omega = Matrix4d::Zero();
  Omega.block<3, 3>(0, 0) = -skewSymmetric(gyro);
  Omega.block<3, 1>(0, 3) = gyro;
  Omega.block<1, 3>(3, 0) = -gyro;

  Vector4d& q = state_server.imu_state.orientation;
  Vector3d& v = state_server.imu_state.velocity;
  Vector3d& p = state_server.imu_state.position;

  // Some pre-calculation
  Vector4d dq_dt, dq_dt2;
  if (gyro_norm > 1e-5) {
    dq_dt = (cos(gyro_norm*dt*0.5)*Matrix4d::Identity() +
      1/gyro_norm*sin(gyro_norm*dt*0.5)*Omega) * q;
    dq_dt2 = (cos(gyro_norm*dt*0.25)*Matrix4d::Identity() +
      1/gyro_norm*sin(gyro_norm*dt*0.25)*Omega) * q;
  }
  //值很小的时候,cos(gyro_norm*dt*0.25)=1貌似可以忽略
  else {
    dq_dt = (Matrix4d::Identity()+0.5*dt*Omega) *
      cos(gyro_norm*dt*0.5) * q;
    dq_dt2 = (Matrix4d::Identity()+0.25*dt*Omega) *
      cos(gyro_norm*dt*0.25) * q;
  }
  Matrix3d dR_dt_transpose = quaternionToRotation(dq_dt).transpose();
  Matrix3d dR_dt2_transpose = quaternionToRotation(dq_dt2).transpose();

  // k1 = f(tn, yn)
  Vector3d k1_v_dot = quaternionToRotation(q).transpose()*acc +
    IMUState::gravity;
  Vector3d k1_p_dot = v;

  // k2 = f(tn+dt/2, yn+k1*dt/2)
  Vector3d k1_v = v + k1_v_dot*dt/2;
  Vector3d k2_v_dot = dR_dt2_transpose*acc +
    IMUState::gravity;
  Vector3d k2_p_dot = k1_v;

  // k3 = f(tn+dt/2, yn+k2*dt/2)
  Vector3d k2_v = v + k2_v_dot*dt/2;
  Vector3d k3_v_dot = dR_dt2_transpose*acc +
    IMUState::gravity;
  Vector3d k3_p_dot = k2_v;

  // k4 = f(tn+dt, yn+k3*dt)
  Vector3d k3_v = v + k3_v_dot*dt;
  Vector3d k4_v_dot = dR_dt_transpose*acc +
    IMUState::gravity;
  Vector3d k4_p_dot = k3_v;

  // yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
  q = dq_dt;
  quaternionNormalize(q);
  v = v + dt/6*(k1_v_dot+2*k2_v_dot+2*k3_v_dot+k4_v_dot);
  p = p + dt/6*(k1_p_dot+2*k2_p_dot+2*k3_p_dot+k4_p_dot);

  return;
}

void MsckfVio::stateAugmentation(const double& time) {

  const Matrix3d& R_i_c = state_server.imu_state.R_imu_cam0;
  const Vector3d& t_c_i = state_server.imu_state.t_cam0_imu;

  // Add a new camera state to the state server.
  //1.由上一步得到的IMU状态计算当前相机状态

  Matrix3d R_w_i = quaternionToRotation(
      state_server.imu_state.orientation);
  Matrix3d R_w_c = R_i_c * R_w_i;
  Vector3d t_c_w = state_server.imu_state.position +
    R_w_i.transpose()*t_c_i;

  //2.相机状态的增广
  state_server.cam_states[state_server.imu_state.id] =
    CAMState(state_server.imu_state.id);
  CAMState& cam_state = state_server.cam_states[
    state_server.imu_state.id];

  cam_state.time = time;
  cam_state.orientation = rotationToQuaternion(R_w_c);
  cam_state.position = t_c_w;

  cam_state.orientation_null = cam_state.orientation;
  cam_state.position_null = cam_state.position;

  // Update the covariance matrix of the state.
  // To simplify computation, the matrix J below is the nontrivial block
  // in Equation (16) in "A Multi-State Constraint Kalman Filter for Vision
  // -aided Inertial Navigation".
  //3.计算相对于以前状态向量的雅可比
  //???
  Matrix<double, 6, 21> J = Matrix<double, 6, 21>::Zero();
  J.block<3, 3>(0, 0) = R_i_c;
  J.block<3, 3>(0, 15) = Matrix3d::Identity();  //???
  J.block<3, 3>(3, 0) = -skewSymmetric(R_w_i.transpose()*t_c_i); //???为了简化计算，使用了msckfmono中的J
  //J.block<3, 3>(3, 0) = -R_w_i.transpose()*skewSymmetric(t_c_i); //???
  J.block<3, 3>(3, 12) = Matrix3d::Identity();
  J.block<3, 3>(3, 18) = Matrix3d::Identity();  //???

  // Resize the state covariance matrix.
  //4.协方差的增广

  int old_rows = state_server.state_cov.rows();
  int old_cols = state_server.state_cov.cols();

  state_server.state_cov.conservativeResize(old_rows+6, old_cols+6);

  // Rename some matrix blocks for convenience.
  const Matrix<double, 21, 21>& P11 =
    state_server.state_cov.block<21, 21>(0, 0);

  const MatrixXd& P12 =
    state_server.state_cov.block(0, 21, 21, old_cols-21);

  // Fill in the augmented state covariance.
  //
  state_server.state_cov.block(old_rows, 0, 6, old_cols) << J*P11, J*P12;
  state_server.state_cov.block(0, old_cols, old_rows, 6) =
    state_server.state_cov.block(old_rows, 0, 6, old_cols).transpose();
  state_server.state_cov.block<6, 6>(old_rows, old_cols) =
    J * P11 * J.transpose();

  // Fix the covariance to be symmetric
  //5.保证协方差的对称性（A+A^T）/2
  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  return;
}
//增加当前帧观测到的点到特征点地图中
void MsckfVio::addFeatureObservations(
    const vector<pair<double, std::vector<Eigen::Matrix<double, 5, 1>>> > & msg) {

  StateIDType state_id = state_server.imu_state.id;
  int curr_feature_num = map_server.size();
  int tracked_feature_num = 0;

  // Add new observations for existing features or new
  // features in the map server.
  for (const auto& feature : msg.begin()->second) {


    if (map_server.find(feature[0]) == map_server.end()) {
      // This is a new feature.
      map_server[feature[0]] = Feature(feature[0]);
      map_server[feature[0]].observations[state_id] =
        Vector4d(feature[1], feature[2],
            feature[3], feature[4]);
    } else {
      // This is an old feature.
      map_server[feature[0]].observations[state_id] =
        Vector4d(feature[1], feature[2],
            feature[3], feature[4]);
      ++tracked_feature_num;
    }
  }

  tracking_rate =
    static_cast<double>(tracked_feature_num) /
    static_cast<double>(curr_feature_num);

  return;
}
//计算每一个特征点对单个相机状态和单个特征点的雅可比
void MsckfVio::measurementJacobian(
    const StateIDType& cam_state_id,
    const FeatureIDType& feature_id,
    Matrix<double, 4, 6>& H_x, Matrix<double, 4, 3>& H_f, Vector4d& r) {

  // Prepare all the required data.
  const CAMState& cam_state = state_server.cam_states[cam_state_id];
  const Feature& feature = map_server[feature_id];

  // Cam0 pose.
  Matrix3d R_w_c0 = quaternionToRotation(cam_state.orientation);
  const Vector3d& t_c0_w = cam_state.position;

  // Cam1 pose.
  Matrix3d R_c0_c1 = CAMState::T_cam0_cam1.linear();
  Matrix3d R_w_c1 = CAMState::T_cam0_cam1.linear() * R_w_c0;
  Vector3d t_c1_w = t_c0_w - R_w_c1.transpose()*CAMState::T_cam0_cam1.translation();

  // 3d feature position in the world frame.
  // And its observation with the stereo cameras.
  const Vector3d& p_w = feature.position;
  const Vector4d& z = feature.observations.find(cam_state_id)->second;

  // Convert the feature position from the world frame to
  // the cam0 and cam1 frame.
  //1.将世界坐标系下的特征点转到相机坐标系
  Vector3d p_c0 = R_w_c0 * (p_w-t_c0_w);
  Vector3d p_c1 = R_w_c1 * (p_w-t_c1_w);

  // Compute the Jacobians.
  //2.计算雅可比
  Matrix<double, 4, 3> dz_dpc0 = Matrix<double, 4, 3>::Zero();
  dz_dpc0(0, 0) = 1 / p_c0(2);
  dz_dpc0(1, 1) = 1 / p_c0(2);
  dz_dpc0(0, 2) = -p_c0(0) / (p_c0(2)*p_c0(2));
  dz_dpc0(1, 2) = -p_c0(1) / (p_c0(2)*p_c0(2));

  Matrix<double, 4, 3> dz_dpc1 = Matrix<double, 4, 3>::Zero();
  dz_dpc1(2, 0) = 1 / p_c1(2);
  dz_dpc1(3, 1) = 1 / p_c1(2);
  dz_dpc1(2, 2) = -p_c1(0) / (p_c1(2)*p_c1(2));
  dz_dpc1(3, 2) = -p_c1(1) / (p_c1(2)*p_c1(2));

  Matrix<double, 3, 6> dpc0_dxc = Matrix<double, 3, 6>::Zero();
  dpc0_dxc.leftCols(3) = skewSymmetric(p_c0);
  dpc0_dxc.rightCols(3) = -R_w_c0;

  Matrix<double, 3, 6> dpc1_dxc = Matrix<double, 3, 6>::Zero();
  dpc1_dxc.leftCols(3) = R_c0_c1 * skewSymmetric(p_c0);
  dpc1_dxc.rightCols(3) = -R_w_c1;

  Matrix3d dpc0_dpg = R_w_c0;
  Matrix3d dpc1_dpg = R_w_c1;

  H_x = dz_dpc0*dpc0_dxc + dz_dpc1*dpc1_dxc;
  H_f = dz_dpc0*dpc0_dpg + dz_dpc1*dpc1_dpg;

  // Modifty the measurement Jacobian to ensure
  // observability constrain.
  //3.改进雅可比，为了能观
  //???
  Matrix<double, 4, 6> A = H_x;
  Matrix<double, 6, 1> u = Matrix<double, 6, 1>::Zero();
  u.block<3, 1>(0, 0) = quaternionToRotation(
      cam_state.orientation_null) * IMUState::gravity;
  u.block<3, 1>(3, 0) = skewSymmetric(
      p_w-cam_state.position_null) * IMUState::gravity;
  H_x = A - A*u*(u.transpose()*u).inverse()*u.transpose();
  H_f = -H_x.block<4, 3>(0, 3);

  // Compute the residual.
  //4.计算残差
  r = z - Vector4d(p_c0(0)/p_c0(2), p_c0(1)/p_c0(2),
      p_c1(0)/p_c1(2), p_c1(1)/p_c1(2));

  return;
}
//计算每一个特征点的雅可比，且映射到左零空间
void MsckfVio::featureJacobian(
    const FeatureIDType& feature_id,
    const std::vector<StateIDType>& cam_state_ids,
    MatrixXd& H_x, VectorXd& r) {

  const auto& feature = map_server[feature_id];

  // Check how many camera states in the provided camera
  // id camera has actually seen this feature.
  //1.检测每个特征点在多少个相机状态中检测到
  vector<StateIDType> valid_cam_state_ids(0);
  for (const auto& cam_id : cam_state_ids) {
    if (feature.observations.find(cam_id) ==
        feature.observations.end()) continue;

    valid_cam_state_ids.push_back(cam_id);
  }

  int jacobian_row_size = 0;
  jacobian_row_size = 4 * valid_cam_state_ids.size();
  //H_xj(4M,21+6N)
  //H_fj(4M,3)
  //r_j(4M,1)
  MatrixXd H_xj = MatrixXd::Zero(jacobian_row_size,
      21+state_server.cam_states.size()*6);
  MatrixXd H_fj = MatrixXd::Zero(jacobian_row_size, 3);
  VectorXd r_j = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;
  //2.计算每个特征点的对应的所有相机状态的雅可比
  for (const auto& cam_id : valid_cam_state_ids) {
    //H_xi(4,6)
    //H_fi(4,3)
    //r_i(4,1)
    Matrix<double, 4, 6> H_xi = Matrix<double, 4, 6>::Zero();
    Matrix<double, 4, 3> H_fi = Matrix<double, 4, 3>::Zero();
    Vector4d r_i = Vector4d::Zero();
    measurementJacobian(cam_id, feature.id, H_xi, H_fi, r_i);

    auto cam_state_iter = state_server.cam_states.find(cam_id);
    int cam_state_cntr = std::distance(
        state_server.cam_states.begin(), cam_state_iter);

    // Stack the Jacobians.
    H_xj.block<4, 6>(stack_cntr, 21+6*cam_state_cntr) = H_xi;
    H_fj.block<4, 3>(stack_cntr, 0) = H_fi;
    r_j.segment<4>(stack_cntr) = r_i;
    stack_cntr += 4;
  }

  // Project the residual and Jacobians onto the nullspace
  // of H_fj.
  //3.映射到左零空间中
  //???
  //ComputeFullU意味着U是方的（4m,4m）的
  JacobiSVD<MatrixXd> svd_helper(H_fj, ComputeFullU | ComputeThinV);
  //A(4m,4m-3)
  MatrixXd A = svd_helper.matrixU().rightCols(
      jacobian_row_size - 3);
  //H_x(4m-3,21+6n)
  //r(4m-3,1)
  H_x = A.transpose() * H_xj;
  r = A.transpose() * r_j;

  return;
}

void MsckfVio::measurementUpdate(
    const MatrixXd& H, const VectorXd& r) {

  if (H.rows() == 0 || r.rows() == 0) return;

  // Decompose the final Jacobian matrix to reduce computational
  // complexity as in Equation (28), (29).
  //H(l*(4M-3),21+6N)
  //r(l*(4M-3),21+6N)
  //1.QR分解
  MatrixXd H_thin;
  VectorXd r_thin;

  if (H.rows() > H.cols()) {
   /*  // Convert H to a sparse matrix.
    SparseMatrix<double> H_sparse = H.sparseView();

    // Perform QR decompostion on H_sparse.
    SPQR<SparseMatrix<double> > spqr_helper;
    spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
    spqr_helper.compute(H_sparse);

    MatrixXd H_temp;
    VectorXd r_temp;
    (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
    (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);
    //H_thin(21+6n,21+6n)
    //r_thin(21+6n,1)
    H_thin = H_temp.topRows(21+state_server.cam_states.size()*6);
    r_thin = r_temp.head(21+state_server.cam_states.size()*6);
    */
    HouseholderQR<MatrixXd> qr_helper(H);
    MatrixXd Q = qr_helper.householderQ();
    MatrixXd Q1 = Q.leftCols(21+state_server.cam_states.size()*6);

    H_thin = Q1.transpose() * H;
    r_thin = Q1.transpose() * r;
  } else {
    H_thin = H;
    r_thin = r;
  }

  // Compute the Kalman gain.
  //2.计算卡尔曼增益
  //最终视觉的噪声协方差变成（Q^T*R*Q）,由于Q是正交矩阵，R是视觉噪声乘以单位阵
  const MatrixXd& P = state_server.state_cov;
  MatrixXd S = H_thin*P*H_thin.transpose() +
      Feature::observation_noise*MatrixXd::Identity(
        H_thin.rows(), H_thin.rows());
  //MatrixXd K_transpose = S.fullPivHouseholderQr().solve(H_thin*P);
  MatrixXd K_transpose = S.ldlt().solve(H_thin*P);
  MatrixXd K = K_transpose.transpose();

  // Compute the error of the state.
  //3.计算状态误差增量
  VectorXd delta_x = K * r_thin;

  // Update the IMU state.
  //4.更新IMU的误差状态
  const VectorXd& delta_x_imu = delta_x.head<21>();

  if (//delta_x_imu.segment<3>(0).norm() > 0.15 ||
      //delta_x_imu.segment<3>(3).norm() > 0.15 ||
      delta_x_imu.segment<3>(6).norm() > 0.5 ||
      //delta_x_imu.segment<3>(9).norm() > 0.5 ||
      delta_x_imu.segment<3>(12).norm() > 1.0) {
    printf("delta velocity: %f\n", delta_x_imu.segment<3>(6).norm());
    printf("delta position: %f\n", delta_x_imu.segment<3>(12).norm());
     std::cout<<"Update change is too large."<<endl;
    //return;
  }
  //
  const Vector4d dq_imu =
    smallAngleQuaternion(delta_x_imu.head<3>());
  state_server.imu_state.orientation = quaternionMultiplication(
      dq_imu, state_server.imu_state.orientation);
  state_server.imu_state.gyro_bias += delta_x_imu.segment<3>(3);
  state_server.imu_state.velocity += delta_x_imu.segment<3>(6);
  state_server.imu_state.acc_bias += delta_x_imu.segment<3>(9);
  state_server.imu_state.position += delta_x_imu.segment<3>(12);

  const Vector4d dq_extrinsic =
    smallAngleQuaternion(delta_x_imu.segment<3>(15));
  state_server.imu_state.R_imu_cam0 = quaternionToRotation(
      dq_extrinsic) * state_server.imu_state.R_imu_cam0;
  state_server.imu_state.t_cam0_imu += delta_x_imu.segment<3>(18);

  // Update the camera states.
  auto cam_state_iter = state_server.cam_states.begin();
  for (int i = 0; i < state_server.cam_states.size();
      ++i, ++cam_state_iter) {
    const VectorXd& delta_x_cam = delta_x.segment<6>(21+i*6);
    const Vector4d dq_cam = smallAngleQuaternion(delta_x_cam.head<3>());
    cam_state_iter->second.orientation = quaternionMultiplication(
        dq_cam, cam_state_iter->second.orientation);
    cam_state_iter->second.position += delta_x_cam.tail<3>();
  }

  // Update state covariance.
  //5.更新协方差矩阵
  MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K*H_thin;
  //state_server.state_cov = I_KH*state_server.state_cov*I_KH.transpose() +
  //  K*K.transpose()*Feature::observation_noise;
  state_server.state_cov = I_KH*state_server.state_cov;

  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  return;
}
//???
bool MsckfVio::gatingTest(
    const MatrixXd& H, const VectorXd& r, const int& dof) {

  MatrixXd P1 = H * state_server.state_cov * H.transpose();
  MatrixXd P2 = Feature::observation_noise *
    MatrixXd::Identity(H.rows(), H.rows());
  double gamma = r.transpose() * (P1+P2).ldlt().solve(r);

  //cout << dof << " " << gamma << " " <<
  //  chi_squared_test_table[dof] << " ";

  if (gamma < chi_squared_test_table[dof]) {
    //cout << "passed" << endl;
    return true;
  } else {
   // cout << "failed" << endl;
    return false;
  }
}

void MsckfVio::removeLostFeatures() {

  // Remove the features that lost track.
  // BTW, find the size the final Jacobian matrix and residual vector.
  int jacobian_row_size = 0;
  vector<FeatureIDType> invalid_feature_ids(0);
  vector<FeatureIDType> processed_feature_ids(0);

  for (auto iter = map_server.begin();
      iter != map_server.end(); ++iter) {
    // Rename the feature to be checked.
    auto& feature = iter->second;
    //1.找到追踪不上的点，且剔除尺寸小于3的点
    // Pass the features that are still being tracked.
    //剔除在当前帧的点
    if (feature.observations.find(state_server.imu_state.id) !=
        feature.observations.end()) continue;
    //剔除尺寸小于3个的
    if (feature.observations.size() < 3) {
      invalid_feature_ids.push_back(feature.id);
      continue;
    }

    // Check if the feature can be initialized if it
    // has not been.
    if (!feature.is_initialized) {
      if (!feature.checkMotion(state_server.cam_states)) {
        invalid_feature_ids.push_back(feature.id);
        continue;
      } else {
        if(!feature.initializePosition(state_server.cam_states)) {
          invalid_feature_ids.push_back(feature.id);
          continue;
        }
      }
    }

    jacobian_row_size += 4*feature.observations.size() - 3;
    processed_feature_ids.push_back(feature.id);
  }

  //cout << "invalid/processed feature #: " <<
  //  invalid_feature_ids.size() << "/" <<
  //  processed_feature_ids.size() << endl;
  //cout << "jacobian row #: " << jacobian_row_size << endl;

  // Remove the features that do not have enough measurements.
  for (const auto& feature_id : invalid_feature_ids)
    map_server.erase(feature_id);

  // Return if there is no lost feature to be processed.
  if (processed_feature_ids.size() == 0) return;
  //左零空间
  //H_x=(l*(4M-3),21+6N),l是特征点的个数，M是每个特征点对应的相机状态个数，N是所有相机状态个数
  //r=(l*(4M-3),1)
  MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
      21+6*state_server.cam_states.size());
  VectorXd r = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  // Process the features which lose track.
  //2.计算所有特征点的雅可比
  for (const auto& feature_id : processed_feature_ids) {
    auto& feature = map_server[feature_id];

    vector<StateIDType> cam_state_ids(0);
    for (const auto& measurement : feature.observations)
      cam_state_ids.push_back(measurement.first);

    MatrixXd H_xj;
    VectorXd r_j;
    //H_xj(4m-3,21+6n)
    //r_j(4m-3,1)
    featureJacobian(feature.id, cam_state_ids, H_xj, r_j);

    if (gatingTest(H_xj, r_j, cam_state_ids.size()-1)) {
      H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
      r.segment(stack_cntr, r_j.rows()) = r_j;
      stack_cntr += H_xj.rows();
    }

    // Put an upper bound on the row size of measurement Jacobian,
    // which helps guarantee the executation time.
    if (stack_cntr > 1500) break;
  }

  H_x.conservativeResize(stack_cntr, H_x.cols());
  r.conservativeResize(stack_cntr);

  // Perform the measurement update step.
  //3.视觉更新
  measurementUpdate(H_x, r);

  // Remove all processed features from the map.
  for (const auto& feature_id : processed_feature_ids)
    map_server.erase(feature_id);

  return;
}
//选择要剔除的哪两个相机状态
void MsckfVio::findRedundantCamStates(
    vector<StateIDType>& rm_cam_state_ids) {

  // Move the iterator to the key position.
  auto key_cam_state_iter = state_server.cam_states.end();//最新帧
  for (int i = 0; i < 4; ++i)
    --key_cam_state_iter;
  auto cam_state_iter = key_cam_state_iter;//第5最新帧
  ++cam_state_iter;//第四最新帧
  auto first_cam_state_iter = state_server.cam_states.begin();//最老帧

  // Pose of the key camera state.
  const Vector3d key_position =
    key_cam_state_iter->second.position;
  const Matrix3d key_rotation = quaternionToRotation(
      key_cam_state_iter->second.orientation);

  // Mark the camera states to be removed based on the
  // motion between states.
  //清除两个相机状态
  //1.如果第五帧与第四帧之间的角度，平移小于阈值以及追踪率大于阈值，则剔除第四帧
  //2.否则就剔除最老帧
  for (int i = 0; i < 2; ++i) {
    const Vector3d position =
      cam_state_iter->second.position;
    const Matrix3d rotation = quaternionToRotation(
        cam_state_iter->second.orientation);

    double distance = (position-key_position).norm();
    double angle = AngleAxisd(
        rotation*key_rotation.transpose()).angle();

    if (angle < rotation_threshold &&
        distance < translation_threshold &&
        tracking_rate > tracking_rate_threshold) {
      rm_cam_state_ids.push_back(cam_state_iter->first);
      ++cam_state_iter;
    } else {
      rm_cam_state_ids.push_back(first_cam_state_iter->first);
      ++first_cam_state_iter;
    }
  }

  // Sort the elements in the output vector.
  sort(rm_cam_state_ids.begin(), rm_cam_state_ids.end());

  return;
}
//如果相机状态超过30个，则剔除两个相机状态
void MsckfVio::pruneCamStateBuffer() {

  if (state_server.cam_states.size() < max_cam_state_size)
    return;

  // Find two camera states to be removed.
  //1.选择要被剔除的两个相机状态
  vector<StateIDType> rm_cam_state_ids(0);
  findRedundantCamStates(rm_cam_state_ids);

  // Find the size of the Jacobian matrix.
  int jacobian_row_size = 0;
  for (auto& item : map_server) {
    auto& feature = item.second;
    // Check how many camera states to be removed are associated
    // with this feature.
    //检查这个点中有几个将要被移除的相机状态
    vector<StateIDType> involved_cam_state_ids(0);
    for (const auto& cam_id : rm_cam_state_ids) {
      if (feature.observations.find(cam_id) !=
          feature.observations.end())
        involved_cam_state_ids.push_back(cam_id);
    }
    //如果该特征点只检测到一个相机状态，直接剔除该特征点
    if (involved_cam_state_ids.size() == 0) continue;
    if (involved_cam_state_ids.size() == 1) {
      feature.observations.erase(involved_cam_state_ids[0]);
      continue;
    }
    //如果特征点没有三角化，则也直接剔除
    if (!feature.is_initialized) {
      // Check if the feature can be initialize.
      if (!feature.checkMotion(state_server.cam_states)) {
        // If the feature cannot be initialized, just remove
        // the observations associated with the camera states
        // to be removed.
        for (const auto& cam_id : involved_cam_state_ids)
          feature.observations.erase(cam_id);
        continue;
      } else {
        if(!feature.initializePosition(state_server.cam_states)) {
          for (const auto& cam_id : involved_cam_state_ids)
            feature.observations.erase(cam_id);
          continue;
        }
      }
    }

    jacobian_row_size += 4*involved_cam_state_ids.size() - 3;
  }

  //cout << "jacobian row #: " << jacobian_row_size << endl;

  // Compute the Jacobian and residual.
  MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
      21+6*state_server.cam_states.size());
  VectorXd r = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  for (auto& item : map_server) {
    auto& feature = item.second;
    // Check how many camera states to be removed are associated
    // with this feature.
    vector<StateIDType> involved_cam_state_ids(0);
    for (const auto& cam_id : rm_cam_state_ids) {
      if (feature.observations.find(cam_id) !=
          feature.observations.end())
        involved_cam_state_ids.push_back(cam_id);
    }

    if (involved_cam_state_ids.size() == 0) continue;

    MatrixXd H_xj;
    VectorXd r_j;
    featureJacobian(feature.id, involved_cam_state_ids, H_xj, r_j);

    if (gatingTest(H_xj, r_j, involved_cam_state_ids.size())) {
      H_x.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
      r.segment(stack_cntr, r_j.rows()) = r_j;
      stack_cntr += H_xj.rows();
    }

    for (const auto& cam_id : involved_cam_state_ids)
      feature.observations.erase(cam_id);
  }

  H_x.conservativeResize(stack_cntr, H_x.cols());
  r.conservativeResize(stack_cntr);

  // Perform measurement update.
  measurementUpdate(H_x, r);

  for (const auto& cam_id : rm_cam_state_ids) {
    int cam_sequence = std::distance(state_server.cam_states.begin(),
        state_server.cam_states.find(cam_id));
    int cam_state_start = 21 + 6*cam_sequence;
    int cam_state_end = cam_state_start + 6;

    // Remove the corresponding rows and columns in the state
    // covariance matrix.
    if (cam_state_end < state_server.state_cov.rows()) {
      state_server.state_cov.block(cam_state_start, 0,
          state_server.state_cov.rows()-cam_state_end,
          state_server.state_cov.cols()) =
        state_server.state_cov.block(cam_state_end, 0,
            state_server.state_cov.rows()-cam_state_end,
            state_server.state_cov.cols());

      state_server.state_cov.block(0, cam_state_start,
          state_server.state_cov.rows(),
          state_server.state_cov.cols()-cam_state_end) =
        state_server.state_cov.block(0, cam_state_end,
            state_server.state_cov.rows(),
            state_server.state_cov.cols()-cam_state_end);

      state_server.state_cov.conservativeResize(
          state_server.state_cov.rows()-6, state_server.state_cov.cols()-6);
    } else {
      state_server.state_cov.conservativeResize(
          state_server.state_cov.rows()-6, state_server.state_cov.cols()-6);
    }

    // Remove this camera state in the state vector.
    state_server.cam_states.erase(cam_id);
  }

  return;
}

void MsckfVio::onlineReset() {

  // Never perform online reset if position std threshold
  // is non-positive.
  //if (position_std_threshold <= 0) return;
  static long long int online_reset_counter = 0;

  // Check the uncertainty of positions to determine if
  // the system can be reset.
  double position_x_std = std::sqrt(state_server.state_cov(12, 12));
  double position_y_std = std::sqrt(state_server.state_cov(13, 13));
  double position_z_std = std::sqrt(state_server.state_cov(14, 14));

  /* if (position_x_std < position_std_threshold &&
      position_y_std < position_std_threshold &&
      position_z_std < position_std_threshold) return;
*/

  // Remove all existing camera states.
  state_server.cam_states.clear();

  // Clear all exsiting features in the map.
  map_server.clear();

  // Reset the state covariance.
  double gyro_bias_cov, acc_bias_cov, velocity_cov;

  velocity_cov=0.25;
  gyro_bias_cov= 1e-4;
  acc_bias_cov=1e-2;

  double extrinsic_rotation_cov, extrinsic_translation_cov;

  extrinsic_rotation_cov=3.0462e-4;
  extrinsic_translation_cov= 1e-4;

  state_server.state_cov = MatrixXd::Zero(21, 21);
  for (int i = 3; i < 6; ++i)
    state_server.state_cov(i, i) = gyro_bias_cov;
  for (int i = 6; i < 9; ++i)
    state_server.state_cov(i, i) = velocity_cov;
  for (int i = 9; i < 12; ++i)
    state_server.state_cov(i, i) = acc_bias_cov;
  for (int i = 15; i < 18; ++i)
    state_server.state_cov(i, i) = extrinsic_rotation_cov;
  for (int i = 18; i < 21; ++i)
    state_server.state_cov(i, i) = extrinsic_translation_cov;

std::cout<<"online reset"<<std::endl;
  return;
}

void MsckfVio::publish(const double& time) {

  // Convert the IMU frame to the body frame.
  const IMUState& imu_state = state_server.imu_state;
  Eigen::Isometry3d T_i_w = Eigen::Isometry3d::Identity();
  T_i_w.linear() = quaternionToRotation(
      imu_state.orientation).transpose();
  T_i_w.translation() = imu_state.position;

  //DEBUG HAN

  std::ofstream foutC(MSCKF_RESULT_PATH, std::ios::app);
  foutC.setf(ios::fixed, ios::floatfield);
  foutC.precision(6);
  foutC << time << " ";
  foutC.precision(5);
  foutC << imu_state.position.x() << " "
        << imu_state.position.y() << " "
        << imu_state.position.z() << " "
        << Quaterniond(quaternionToRotation(imu_state.orientation).transpose()).x() << " "
        << Quaterniond(quaternionToRotation(imu_state.orientation).transpose()).y() << " "
        << Quaterniond(quaternionToRotation(imu_state.orientation).transpose()).z() << " "
        << Quaterniond(quaternionToRotation(imu_state.orientation).transpose()).w() //<< " "
        //<< estimator.Vs[WINDOW_SIZE].x() << ","
        //<< estimator.Vs[WINDOW_SIZE].y() << ","
        //<< estimator.Vs[WINDOW_SIZE].z() << ","
        << endl;
  foutC.close();
  std::cout.setf(std::ios::fixed, std::ios::floatfield);
  std::cout.precision(6);
  std::cout<<"time"<<time<<"translation"<<imu_state.position.x() << " "
        << imu_state.position.y() << " "
        << imu_state.position.z() << " "
        <<"rotation"
        << Quaterniond(quaternionToRotation(imu_state.orientation).transpose()).x() << " "
        << Quaterniond(quaternionToRotation(imu_state.orientation).transpose()).y() << " "
        << Quaterniond(quaternionToRotation(imu_state.orientation).transpose()).z() << " "
        << Quaterniond(quaternionToRotation(imu_state.orientation).transpose()).w()<<std::endl;
  return;
}

} // namespace msckf_vio

