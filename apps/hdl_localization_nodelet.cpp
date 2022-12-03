#include <mutex>
#include <memory>
#include <iostream>

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <eigen_conversions/eigen_msg.h>

#include <std_srvs/Empty.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <tf/transform_broadcaster.h>

#include <pcl/filters/voxel_grid.h>

#include <pclomp/ndt_omp.h>
#include <fast_gicp/ndt/ndt_cuda.hpp>

#include <hdl_localization/pose_estimator.hpp>
#include <hdl_localization/delta_estimater.hpp>

#include <hdl_localization/ScanMatchingStatus.h>
#include <hdl_global_localization/SetGlobalMap.h>
#include <hdl_global_localization/QueryGlobalLocalization.h>

namespace hdl_localization {

class HdlLocalizationNodelet : public nodelet::Nodelet {
public:
  using PointT = pcl::PointXYZI;

  HdlLocalizationNodelet() : tf_buffer(), tf_listener(tf_buffer) {
  }
  virtual ~HdlLocalizationNodelet() {
  }

  void onInit() override {
    nh = getNodeHandle();
    mt_nh = getMTNodeHandle();
    private_nh = getPrivateNodeHandle();

    initialize_params();

    robot_odom_frame_id = private_nh.param<std::string>("robot_odom_frame_id", "robot_odom");
    odom_child_frame_id = private_nh.param<std::string>("odom_child_frame_id", "base_link");

    use_imu = private_nh.param<bool>("use_imu", true);
    invert_acc = private_nh.param<bool>("invert_acc", false);
    invert_gyro = private_nh.param<bool>("invert_gyro", false);
    if (use_imu) {
      NODELET_INFO("enable imu-based prediction");
      imu_sub = mt_nh.subscribe("/gpsimu_driver/imu_data", 256, &HdlLocalizationNodelet::imu_callback, this);
    }
    points_sub = mt_nh.subscribe("/velodyne_points", 5, &HdlLocalizationNodelet::points_callback, this);
    globalmap_sub = nh.subscribe("/globalmap", 1, &HdlLocalizationNodelet::globalmap_callback, this);
    initialpose_sub = nh.subscribe("/initialpose", 8, &HdlLocalizationNodelet::initialpose_callback, this);

    pose_pub = nh.advertise<nav_msgs::Odometry>("/odom", 5, false);
    aligned_pub = nh.advertise<sensor_msgs::PointCloud2>("/aligned_points", 5, false);
    status_pub = nh.advertise<ScanMatchingStatus>("/status", 5, false);

    // global localization
    use_global_localization = private_nh.param<bool>("use_global_localization", true);
    if(use_global_localization) {
      NODELET_INFO_STREAM("wait for global localization services");
      ros::service::waitForService("/hdl_global_localization/set_global_map");
      ros::service::waitForService("/hdl_global_localization/query");

      set_global_map_service = nh.serviceClient<hdl_global_localization::SetGlobalMap>("/hdl_global_localization/set_global_map");
      query_global_localization_service = nh.serviceClient<hdl_global_localization::QueryGlobalLocalization>("/hdl_global_localization/query");

      relocalize_server = nh.advertiseService("/relocalize", &HdlLocalizationNodelet::relocalize, this);
    }
  }

private:
  pcl::Registration<PointT, PointT>::Ptr create_registration() const {
    std::string reg_method = private_nh.param<std::string>("reg_method", "NDT_OMP");
    std::string ndt_neighbor_search_method = private_nh.param<std::string>("ndt_neighbor_search_method", "DIRECT7");
    double ndt_neighbor_search_radius = private_nh.param<double>("ndt_neighbor_search_radius", 2.0);
    double ndt_resolution = private_nh.param<double>("ndt_resolution", 1.0);

    if(reg_method == "NDT_OMP") {
      NODELET_INFO("NDT_OMP is selected");
      pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
      ndt->setTransformationEpsilon(0.01);
      ndt->setResolution(ndt_resolution);
      if (ndt_neighbor_search_method == "DIRECT1") {
        NODELET_INFO("search_method DIRECT1 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        NODELET_INFO("search_method DIRECT7 is selected");
        ndt->setNeighborhoodSearchMethod(pclomp::DIRECT7);
      } else {
        if (ndt_neighbor_search_method == "KDTREE") {
          NODELET_INFO("search_method KDTREE is selected");
        } else {
          NODELET_WARN("invalid search method was given");
          NODELET_WARN("default method is selected (KDTREE)");
        }
        ndt->setNeighborhoodSearchMethod(pclomp::KDTREE);
      }
      return ndt;
    } else if(reg_method.find("NDT_CUDA") != std::string::npos) {
      NODELET_INFO("NDT_CUDA is selected");
      boost::shared_ptr<fast_gicp::NDTCuda<PointT, PointT>> ndt(new fast_gicp::NDTCuda<PointT, PointT>);
      ndt->setResolution(ndt_resolution);

      if(reg_method.find("D2D") != std::string::npos) {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::D2D);
      } else if (reg_method.find("P2D") != std::string::npos) {
        ndt->setDistanceMode(fast_gicp::NDTDistanceMode::P2D);
      }

      if (ndt_neighbor_search_method == "DIRECT1") {
        NODELET_INFO("search_method DIRECT1 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT1);
      } else if (ndt_neighbor_search_method == "DIRECT7") {
        NODELET_INFO("search_method DIRECT7 is selected");
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT7);
      } else if (ndt_neighbor_search_method == "DIRECT_RADIUS") {
        NODELET_INFO_STREAM("search_method DIRECT_RADIUS is selected : " << ndt_neighbor_search_radius);
        ndt->setNeighborSearchMethod(fast_gicp::NeighborSearchMethod::DIRECT_RADIUS, ndt_neighbor_search_radius);
      } else {
        NODELET_WARN("invalid search method was given");
      }
      return ndt;
    }

    NODELET_ERROR_STREAM("unknown registration method:" << reg_method);
    return nullptr;
  }

  void initialize_params() {
    // intialize scan matching method
    double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    downsample_filter = voxelgrid;

    NODELET_INFO("create registration method for localization");
    registration = create_registration();

    // global localization
    NODELET_INFO("create registration method for fallback during relocalization");
    relocalizing = false;
    delta_estimater.reset(new DeltaEstimater(create_registration()));

    // initialize pose estimator
    if(private_nh.param<bool>("specify_init_pose", true)) {
      NODELET_INFO("initialize pose estimator with specified parameters!!");
      pose_estimator.reset(new hdl_localization::PoseEstimator(registration,
        ros::Time::now(),
        Eigen::Vector3f(private_nh.param<double>("init_pos_x", 0.0), private_nh.param<double>("init_pos_y", 0.0), private_nh.param<double>("init_pos_z", 0.0)),
        Eigen::Quaternionf(private_nh.param<double>("init_ori_w", 1.0), private_nh.param<double>("init_ori_x", 0.0), private_nh.param<double>("init_ori_y", 0.0), private_nh.param<double>("init_ori_z", 0.0)),
        private_nh.param<double>("cool_time_duration", 0.5)
      ));
    }
  }

private:
  /**
   * @brief callback for imu data
   * @param imu_msg
   */
  void imu_callback(const sensor_msgs::ImuConstPtr& imu_msg) {
    std::lock_guard<std::mutex> lock(imu_data_mutex);
    imu_data.push_back(imu_msg);
  }

  /**
   * @brief callback for point cloud data
   * @param points_msg
   */
  void points_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg) {

    static tf::TransformBroadcaster br;
    static const Eigen::Vector3d Lidar_offset_to_IMU(0.05512, 0.02226, 0.0297); // Horizon

    tf::Transform transform;
    tf::Quaternion q;
    transform.setOrigin(tf::Vector3(Lidar_offset_to_IMU[0], Lidar_offset_to_IMU[1], Lidar_offset_to_IMU[2]));
    q.setW(1);
    q.setX(0);
    q.setY(0);
    q.setZ(0);
    transform.setRotation(q);
    br.sendTransform(tf::StampedTransform(transform, points_msg->header.stamp, "body", "livox"));

    std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex);
    if(!pose_estimator) {
      NODELET_ERROR("waiting for initial pose input!!");
      return;
    }

    if(!globalmap) {
      NODELET_ERROR("globalmap has not been received!!");
      return;
    }

    const auto& stamp = points_msg->header.stamp;
    pcl::PointCloud<PointT>::Ptr pcl_cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *pcl_cloud);

    if(pcl_cloud->empty()) {
      NODELET_ERROR("cloud is empty!!");
      return;
    }

    // transform pointcloud into odom_child_frame_id
    std::string tfError;
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());

    // 转换失败，直接跳过
    // 不用里程计，直接用livox的frame
    // modify by ln 20221122
    // *cloud = *pcl_cloud;
    try
    {
      if( this->tf_buffer.canTransform(odom_child_frame_id, pcl_cloud->header.frame_id, stamp, ros::Duration(0.2), &tfError) )
      {
          // std::cout << __FILE__ << ":" << __LINE__ << odom_child_frame_id << "--" << pcl_cloud->header.frame_id << std::endl;  
          geometry_msgs::TransformStamped body_pose_in_map = tf_buffer.lookupTransform(odom_child_frame_id, pcl_cloud->header.frame_id, stamp, ros::Duration(0.1));
          // std::cout << body_pose_in_map << std::endl;
          // tf2::Transform tt;
          // tf2::fromMsg(body_pose_in_map, tt);
          if(!pcl_ros::transformPointCloud(odom_child_frame_id, *pcl_cloud, *cloud, this->tf_buffer)) {
              // std::cout << __FILE__ << ":" << __LINE__ << "---------------------------------" << std::endl;
              ROS_ERROR("point cloud cannot be transformed into target frame!!");
              return;
          }
      }
      else
      {
        ROS_ERROR("cannot be transformed");
        // return;
      }
    }
    catch(tf::TransformException &ex)
    {
      ROS_WARN("%s", ex.what());
      NODELET_ERROR(tfError.c_str());
      return;
    }

    // // pcl::PointCloud<PointT>::Ptr temp(new pcl::PointCloud<PointT>());
    // *cloud = *pcl_cloud;
    
    // cloud->points.clear();
    // std::cout << __FILE__ << ":" << __LINE__ << " :" << cloud->points.size()  << std::endl;

    //   cloud->header.frame_id = "base_link";
    //   cloud->header.stamp = pcl_cloud->header.stamp;

    // for(int i=0; i < pcl_cloud->points.size() ; i++)
    // {
    //   PointT pp;
    //   pp.x = pcl_cloud->points[i].x + Lidar_offset_to_IMU[0];
    //   pp.y = pcl_cloud->points[i].y + Lidar_offset_to_IMU[1];
    //   pp.z = pcl_cloud->points[i].z + Lidar_offset_to_IMU[2];      
    //   cloud->points.push_back(pp);
    // }

    // pcl::io::savePCDFile ( "/home/liunao/hdl/" + pcl_cloud->header.frame_id + ".pcd", *pcl_cloud );
    // pcl::io::savePCDFile ( "/home/liunao/hdl/base_link.pcd" , *temp );
    // aligned_pub.publish(temp);
    // return;

    cloud->header.frame_id = odom_child_frame_id;
    // std::cout << __FILE__ << ":" << __LINE__ << " :" << cloud->points.size()  << std::endl;
    if (cloud->header.frame_id.empty())
    {
      std::cout << __FILE__ << ":" << __LINE__ << "---------------------------------" << odom_child_frame_id << "--" << pcl_cloud->header.frame_id << std::endl;  
      ROS_ERROR("cloud->header.frame_id.isempty");
      // return ;
    }
    // 点云已经是提取的 surf 点，不滤波
    // auto filtered = downsample(cloud);
    auto filtered = cloud;
    filtered->header = cloud->header;
    if (filtered->header.frame_id.empty())
    {
      ROS_ERROR("filtered->header.frame_id.isempty");
      // return ;
    }
    else
    {
      // std::cout << __FILE__ << ":" << __LINE__ << "--- filtered->header.frame_id : " << filtered->header.frame_id <<  filtered->points.size()  << std::endl;
    }

    last_scan = filtered;

    if(relocalizing) {
      delta_estimater->add_frame(filtered);
    }

    Eigen::Matrix4f before = pose_estimator->matrix();

    // predict
    if(!use_imu) {
      pose_estimator->predict(stamp);
    } else {
      // std::cout << __FILE__ << ":" << __LINE__ << "  use imu " << std::endl;
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      auto imu_iter = imu_data.begin();
      for(imu_iter; imu_iter != imu_data.end(); imu_iter++) {
        if(stamp < (*imu_iter)->header.stamp) {
          break;
        }
        const auto& acc = (*imu_iter)->linear_acceleration;
        const auto& gyro = (*imu_iter)->angular_velocity;
        double acc_sign = invert_acc ? -1.0 : 1.0;
        double gyro_sign = invert_gyro ? -1.0 : 1.0;
        pose_estimator->predict((*imu_iter)->header.stamp, acc_sign * Eigen::Vector3f(acc.x, acc.y, acc.z), gyro_sign * Eigen::Vector3f(gyro.x, gyro.y, gyro.z));
      }
      imu_data.erase(imu_data.begin(), imu_iter);
    }

    // odometry-based prediction
    ros::Time last_correction_time = pose_estimator->last_correction_time();
    if(private_nh.param<bool>("enable_robot_odometry_prediction", false) && !last_correction_time.isZero())
    {
      geometry_msgs::TransformStamped odom_delta;
      if(tf_buffer.canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, ros::Duration(0.05))) {
        odom_delta = tf_buffer.lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, ros::Duration(0));
      } else if(tf_buffer.canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, ros::Time(0), robot_odom_frame_id, ros::Duration(0))) {
        odom_delta = tf_buffer.lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, ros::Time(0), robot_odom_frame_id, ros::Duration(0));
      }

      if(odom_delta.header.stamp.isZero()) {
        NODELET_WARN_STREAM("failed to look up transform between " << cloud->header.frame_id << " and " << robot_odom_frame_id);
      } else {
        Eigen::Isometry3d delta = tf2::transformToEigen(odom_delta);
        pose_estimator->predict_odom(delta.cast<float>().matrix());
        // std::cout << __FILE__ << ":" << __LINE__ << ". enable_robot_odometry_prediction true " << std::endl;
      }
    }

    // correct
    auto aligned = pose_estimator->correct(stamp, filtered);

    if(aligned_pub.getNumSubscribers()) {
      aligned->header.frame_id = "map";
      aligned->header.stamp = cloud->header.stamp;
      aligned_pub.publish(aligned);
    }

    publish_scan_matching_status(points_msg->header, aligned);

    publish_odometry(points_msg->header.stamp, pose_estimator->matrix());

  }

  /**
   * @brief callback for globalmap input
   * @param points_msg
   */
  void globalmap_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg) {
    NODELET_INFO("globalmap received!");
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *cloud);
    globalmap = cloud;

    registration->setInputTarget(globalmap);

    if(use_global_localization) {
      NODELET_INFO("set globalmap for global localization!");
      hdl_global_localization::SetGlobalMap srv;
      pcl::toROSMsg(*globalmap, srv.request.global_map);

      if(!set_global_map_service.call(srv)) {
        NODELET_INFO("failed to set global map");
      } else {
        NODELET_INFO("done");
      }
    }
  }

  /**
   * @brief perform global localization to relocalize the sensor position
   * @param
   */
  bool relocalize(std_srvs::EmptyRequest& req, std_srvs::EmptyResponse& res) {
    // ROS_ERROR("Try global localization.");
    if(last_scan == nullptr) {
      NODELET_INFO_STREAM("no scan has been received");
      return false;
    }

    relocalizing = true;
    delta_estimater->reset();
    pcl::PointCloud<PointT>::ConstPtr scan = last_scan;

    hdl_global_localization::QueryGlobalLocalization srv;
    pcl::toROSMsg(*scan, srv.request.cloud);
    srv.request.max_num_candidates = 1;

    if(!query_global_localization_service.call(srv) || srv.response.poses.empty()) {
      relocalizing = false;
      NODELET_INFO_STREAM("global localization failed");
      return false;
    }

    const auto& result = srv.response.poses[0];

    NODELET_INFO_STREAM("--- Global localization result ---");
    NODELET_INFO_STREAM("Trans :" << result.position.x << " " << result.position.y << " " << result.position.z);
    NODELET_INFO_STREAM("Quat  :" << result.orientation.x << " " << result.orientation.y << " " << result.orientation.z << " " << result.orientation.w);
    NODELET_INFO_STREAM("Error :" << srv.response.errors[0]);
    NODELET_INFO_STREAM("Inlier:" << srv.response.inlier_fractions[0]);

    Eigen::Isometry3f pose = Eigen::Isometry3f::Identity();
    pose.linear() = Eigen::Quaternionf(result.orientation.w, result.orientation.x, result.orientation.y, result.orientation.z).toRotationMatrix();
    pose.translation() = Eigen::Vector3f(result.position.x, result.position.y, result.position.z);
    pose = pose * delta_estimater->estimated_delta();

    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    pose_estimator.reset(new hdl_localization::PoseEstimator(
      registration,
      ros::Time::now(),
      pose.translation(),
      Eigen::Quaternionf(pose.linear()),
      private_nh.param<double>("cool_time_duration", 0.5)));

    relocalizing = false;

    return true;
  }

  /**
   * @brief callback for initial pose input ("2D Pose Estimate" on rviz)
   * @param pose_msg
   */
  void initialpose_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg) {
    NODELET_INFO("initial pose received!!");
    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    const auto& p = pose_msg->pose.pose.position;
    const auto& q = pose_msg->pose.pose.orientation;
    pose_estimator.reset(
          new hdl_localization::PoseEstimator(
            registration,
            ros::Time::now(),
            Eigen::Vector3f(p.x, p.y, p.z),
            Eigen::Quaternionf(q.w, q.x, q.y, q.z),
            private_nh.param<double>("cool_time_duration", 0.5))
    );
  }

  /**
   * @brief downsampling
   * @param cloud   input cloud
   * @return downsampled cloud
   */
  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if(!downsample_filter) {
      return cloud;
    }

    pcl::PointCloud<PointT>::Ptr filtered(new pcl::PointCloud<PointT>());
    downsample_filter->setInputCloud(cloud);
    downsample_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  /**
   * @brief publish odometry
   * @param stamp  timestamp
   * @param pose   odometry pose to be published
   */
  void publish_odometry(const ros::Time& stamp, const Eigen::Matrix4f& pose) {
    // broadcast the transform over tf
    if(tf_buffer.canTransform(robot_odom_frame_id, odom_child_frame_id, ros::Time(0))) {
      geometry_msgs::TransformStamped map_wrt_frame = tf2::eigenToTransform(Eigen::Isometry3d(pose.inverse().cast<double>()));
      map_wrt_frame.header.stamp = stamp;
      map_wrt_frame.header.frame_id = odom_child_frame_id;
      map_wrt_frame.child_frame_id = "map";

      geometry_msgs::TransformStamped frame_wrt_odom = tf_buffer.lookupTransform(robot_odom_frame_id, odom_child_frame_id, ros::Time(0), ros::Duration(1));
      // geometry_msgs::TransformStamped frame_wrt_odom = tf_buffer.lookupTransform(robot_odom_frame_id, odom_child_frame_id, stamp, ros::Duration(0.1));
      
      // std::cout << "frame_wrt_odom:" << frame_wrt_odom << std::endl;

      Eigen::Matrix4f frame2odom = tf2::transformToEigen(frame_wrt_odom).cast<float>().matrix();

      geometry_msgs::TransformStamped map_wrt_odom;
      tf2::doTransform(map_wrt_frame, map_wrt_odom, frame_wrt_odom);

      tf2::Transform odom_wrt_map;
      tf2::fromMsg(map_wrt_odom.transform, odom_wrt_map);
      odom_wrt_map = odom_wrt_map.inverse();

      geometry_msgs::TransformStamped odom_trans;
      odom_trans.transform = tf2::toMsg(odom_wrt_map);
      odom_trans.header.stamp = stamp;
      odom_trans.header.frame_id = "map";
      odom_trans.child_frame_id = robot_odom_frame_id;

      // std::cout << __FILE__ << ":" << __LINE__ << "-" << robot_odom_frame_id << "--" <<  << std::endl;
      ROS_INFO("%s,  %s",robot_odom_frame_id.c_str() , odom_child_frame_id.c_str());

      tf_broadcaster.sendTransform(odom_trans);
    } else {
      // std::cout << __FILE__ << ":" << __LINE__ << "------" << std::endl;
      ROS_WARN("%s,  %s",robot_odom_frame_id.c_str() , odom_child_frame_id.c_str());
      geometry_msgs::TransformStamped odom_trans = tf2::eigenToTransform(Eigen::Isometry3d(pose.cast<double>()));
      odom_trans.header.stamp = stamp;
      odom_trans.header.frame_id = "map";
      odom_trans.child_frame_id = odom_child_frame_id;
      tf_broadcaster.sendTransform(odom_trans);
    }

    // publish the transform
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "map";
    odom.twist.twist.linear.x = 0.0;
    odom.twist.twist.linear.y = 0.0;
    odom.twist.twist.angular.z = 0.0;

    // tf::poseEigenToMsg(Eigen::Isometry3d(pose.cast<double>()), odom.pose.pose);
    // odom.child_frame_id = odom_child_frame_id;
    // pose_pub.publish(odom);

    // printf("%f %f %f %f %f %f %f %f \n", stamp.toSec(), odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z
    // , odom.pose.pose.orientation.x  , odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w );

    static std::ofstream ofs;
    ofs.open( "/home/liunao/hdl/hdl_pose.txt", std::ios::app);
    ofs << std::to_string(stamp.toSec())
        << " "  << std::to_string(odom.pose.pose.position.x)
        << " "  << std::to_string(odom.pose.pose.position.y)
        << " "  << std::to_string(odom.pose.pose.position.z)
        << " "  << std::to_string(odom.pose.pose.orientation.x)
        << " "  << std::to_string(odom.pose.pose.orientation.y)
        << " "  << std::to_string(odom.pose.pose.orientation.z)
        << " "  << std::to_string(odom.pose.pose.orientation.w) << std::endl;
    ofs.close();

    std::string tfError;
    try
    {      
      if( this->tf_buffer.canTransform("map", "body" , stamp, ros::Duration(0.05), &tfError) )
      {
        geometry_msgs::TransformStamped body_pose_in_map = tf_buffer.lookupTransform("map", "body", stamp, ros::Duration(0.05));
        tf::poseEigenToMsg(Eigen::Isometry3d(body_pose_in_map.cast<double>()), odom.pose.pose);
        odom.child_frame_id = "body";
        pose_pub.publish(odom);

        static std::ofstream ofs1;
        ofs1.open( "/home/liunao/hdl/hdl_body_pose.txt", std::ios::app);
    
        ofs1 << std::to_string(body_pose_in_map.header.stamp.toSec())
            << " "  << std::to_string(body_pose_in_map.transform.translation.x)
            << " "  << std::to_string(body_pose_in_map.transform.translation.y)
            << " "  << std::to_string(body_pose_in_map.transform.translation.z)
            << " "  << std::to_string(body_pose_in_map.transform.rotation.x)
            << " "  << std::to_string(body_pose_in_map.transform.rotation.y)
            << " "  << std::to_string(body_pose_in_map.transform.rotation.z)
            << " "  << std::to_string(body_pose_in_map.transform.rotation.w) << std::endl;
        ofs1.close();
      }
      else
      {
        ROS_WARN(" can not canTransform ." );
      }
    }
    catch(tf::TransformException &ex)
    {
      ROS_WARN("%s", ex.what());
      NODELET_ERROR(tfError.c_str());
      return;
    }
    // ROS_ERROR("11");
  }

  /**
   * @brief publish scan matching status information
   */
  void publish_scan_matching_status(const std_msgs::Header& header, pcl::PointCloud<pcl::PointXYZI>::ConstPtr aligned) {
    ScanMatchingStatus status;
    status.header = header;

    status.has_converged = registration->hasConverged();
    status.matching_error = registration->getFitnessScore();
    // ROS_INFO("has_converged is %d, matching_error(getFitnessScore) is %f", status.has_converged, status.matching_error);

    double max_correspondence_dist = 0.25;
    max_correspondence_dist = private_nh.param<double>("/max_correspondence_dist", 0.25);

    int num_inliers = 0;
    std::vector<int> k_indices;
    std::vector<float> k_sq_dists;
    for(int i = 0; i < aligned->size(); i++) {
      const auto& pt = aligned->at(i);
      registration->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
      if(k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
        num_inliers++;
      }
    }
    status.inlier_fraction = static_cast<float>(num_inliers) / aligned->size();
    status.relative_pose = tf2::eigenToTransform(Eigen::Isometry3d(registration->getFinalTransformation().cast<double>())).transform;

    // 匹配好的点<内点比例> 大于 该阈值认为 定位成功
    float LOC_SUCCESS_THRESHLOD = 0.7;
    LOC_SUCCESS_THRESHLOD = private_nh.param<double>("/LOC_SUCCESS_THRESHLOD", 0.7);
      // publish_odometry(header.stamp, registration->getFinalTransformation() );
    // if( status.inlier_fraction > LOC_SUCCESS_THRESHLOD )
    // {
    // }
    // else
    // {
    //   ROS_WARN("inlier_fraction is<%lf> less LOC_SUCCESS_THRESHLOD : %lf .max_correspondence_dist is : %lf", status.inlier_fraction, LOC_SUCCESS_THRESHLOD, max_correspondence_dist);
    //   // TODO call relocate locate falied .try to call /relocate service .
    //   return ;
    // }

    status.prediction_labels.reserve(2);
    status.prediction_errors.reserve(2);

    std::vector<double> errors(6, 0.0);

    if(pose_estimator->wo_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = "without_pred";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->wo_prediction_error().get().cast<double>())).transform);
    }

    if(pose_estimator->imu_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = use_imu ? "imu" : "motion_model";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->imu_prediction_error().get().cast<double>())).transform);
    }

    if(pose_estimator->odom_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = "odom";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->odom_prediction_error().get().cast<double>())).transform);
    }

    status_pub.publish(status);
  }

private:
  // ROS
  ros::NodeHandle nh;
  ros::NodeHandle mt_nh;
  ros::NodeHandle private_nh;

  std::string robot_odom_frame_id;
  std::string odom_child_frame_id;

  bool use_imu;
  bool invert_acc;
  bool invert_gyro;
  ros::Subscriber imu_sub;
  ros::Subscriber points_sub;
  ros::Subscriber globalmap_sub;
  ros::Subscriber initialpose_sub;

  ros::Publisher pose_pub;
  ros::Publisher aligned_pub;
  ros::Publisher status_pub;

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener;
  tf2_ros::TransformBroadcaster tf_broadcaster;

  // imu input buffer
  std::mutex imu_data_mutex;
  std::vector<sensor_msgs::ImuConstPtr> imu_data;

  // globalmap and registration method
  pcl::PointCloud<PointT>::Ptr globalmap;
  pcl::Filter<PointT>::Ptr downsample_filter;
  pcl::Registration<PointT, PointT>::Ptr registration;

  // pose estimator
  std::mutex pose_estimator_mutex;
  std::unique_ptr<hdl_localization::PoseEstimator> pose_estimator;

  // global localization
  bool use_global_localization;
  std::atomic_bool relocalizing;
  std::unique_ptr<DeltaEstimater> delta_estimater;

  pcl::PointCloud<PointT>::ConstPtr last_scan;
  ros::ServiceServer relocalize_server;
  ros::ServiceClient set_global_map_service;
  ros::ServiceClient query_global_localization_service;
};
}


PLUGINLIB_EXPORT_CLASS(hdl_localization::HdlLocalizationNodelet, nodelet::Nodelet)
