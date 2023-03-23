#include <mutex>
#include <memory>
#include <iostream>
#include <iomanip>

#include <ros/ros.h>

#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/convert.h>
#include <eigen_conversions/eigen_msg.h>
// #include <Eigen/Geometry.h>

#include <std_msgs/Float32.h>
#include <std_srvs/Empty.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <tf/transform_broadcaster.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>

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

  HdlLocalizationNodelet() : tf_buffer(), tf_listener(tf_buffer) {}
  virtual ~HdlLocalizationNodelet() {}

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
    points_sub = mt_nh.subscribe("/velodyne_points", 1, &HdlLocalizationNodelet::points_callback, this);
    globalmap_sub = nh.subscribe("/globalmap", 1, &HdlLocalizationNodelet::globalmap_callback, this);
    initialpose_sub = nh.subscribe("/initialpose", 8, &HdlLocalizationNodelet::initialpose_callback, this);
    odom_sub = nh.subscribe("/r2live/odometry", 8, &HdlLocalizationNodelet::odom_callback, this);

    pose_pub = nh.advertise<nav_msgs::Odometry>("odom", 1, false);
    aligned_pub = nh.advertise<sensor_msgs::PointCloud2>("aligned_points", 1, false);
    status_pub = nh.advertise<ScanMatchingStatus>("/status", 5, false);
    confidence_pub = nh.advertise<std_msgs::Float32>("/confidece", 5, false);
    string_pub = nh.advertise<std_msgs::String>("/XY_yaw", 5, false);

    // global localization
    use_global_localization = private_nh.param<bool>("use_global_localization", true);
    if (use_global_localization) {
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
    NODELET_WARN("ndt_resolution is %lf", ndt_resolution);

    if (reg_method == "NDT_OMP") {
      NODELET_INFO("NDT_OMP is selected");
      pclomp::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pclomp::NormalDistributionsTransform<PointT, PointT>());
      ndt->setTransformationEpsilon(0.01);
      ndt->setResolution(ndt_resolution);

      ndt->setNumThreads(10);
      double ndt_setStepSize = private_nh.param<double>("ndt_stepsize", 0.1);  // 默认0.1
      ndt->setStepSize(ndt_setStepSize);

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
    } else if (reg_method.find("NDT_CUDA") != std::string::npos) {
      NODELET_INFO("NDT_CUDA is selected");
      boost::shared_ptr<fast_gicp::NDTCuda<PointT, PointT>> ndt(new fast_gicp::NDTCuda<PointT, PointT>);
      ndt->setResolution(ndt_resolution);

      if (reg_method.find("D2D") != std::string::npos) {
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

    no_odom_flag = false;

    // global localization
    NODELET_INFO("create registration method for fallback during relocalization");
    relocalizing = false;
    delta_estimater.reset(new DeltaEstimater(create_registration()));

    // initialize pose estimator
    if (private_nh.param<bool>("specify_init_pose", true)) {
      NODELET_INFO("initialize pose estimator with specified parameters!!");
      pose_estimator.reset(new hdl_localization::PoseEstimator(
        registration,
        ros::Time::now(),
        Eigen::Vector3f(private_nh.param<double>("init_pos_x", 0.0), private_nh.param<double>("init_pos_y", 0.0), private_nh.param<double>("init_pos_z", 0.0)),
        Eigen::Quaternionf(
          private_nh.param<double>("init_ori_w", 1.0),
          private_nh.param<double>("init_ori_x", 0.0),
          private_nh.param<double>("init_ori_y", 0.0),
          private_nh.param<double>("init_ori_z", 0.0)),
        private_nh.param<double>("cool_time_duration", 0.5)));
    }
  }

private:
  /**
   * @brief callback for odom pose input
   * @param odom_msg
   */
  void odom_callback(const nav_msgs::OdometryConstPtr& odom_msg) {
    // NODELET_INFO("odom_msg pose received!!");
    std::lock_guard<std::mutex> lock(odom_data_mutex);
    // odom_data.push_back(odom_msg);
    odom_data.emplace_back(odom_msg);

    // hdl is ok long time, this vector will  increase always. so delete some data
    if (odom_data.size() > 30) {
      // ROS_WARN(" odom_data.size() > 30. delete front 15 old data :");
      odom_data.erase(odom_data.begin(), odom_data.begin() + 15);
    }
  }

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
    // static int pc_cnts = 0;
    // if (pc_cnts++ % 2) {
    //   return;
    // }

    ros::Time s_t = ros::Time::now();

    std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex);
    if (!pose_estimator) {
      NODELET_ERROR("waiting for initial pose input!!");
      return;
    }

    if (!globalmap) {
      NODELET_ERROR("globalmap has not been received!!");
      return;
    }

    const auto& stamp = points_msg->header.stamp;
    pcl::PointCloud<PointT>::Ptr pcl_cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *pcl_cloud);

    if (pcl_cloud->empty()) {
      NODELET_ERROR("cloud is empty!!");
      return;
    }
    
    // ROS_WARN(" pc size() %d ", pcl_cloud->size());
    static const float range = 1.0;
    pcl::CropBox< PointT > cropBoxFilter (true);
    cropBoxFilter.setInputCloud (pcl_cloud);
    cropBoxFilter.setMin (Eigen::Vector4f  (-range, -range, -range, 1.0f));
    cropBoxFilter.setMax (Eigen::Vector4f  (range, range, range, 1.0f));
    cropBoxFilter.setNegative(true);
    cropBoxFilter.filter (*pcl_cloud);
    // ROS_WARN(" pc size() %d ", pcl_cloud->size());

    // transform pointcloud into odom_child_frame_id
    std::string tfError;
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());

    // 转换失败，直接跳过
    // 不用里程计，直接用livox的frame
    // modify by ln 20221122
    // *cloud = *pcl_cloud;
    /*         */
    try {
      if (this->tf_buffer.canTransform(odom_child_frame_id, pcl_cloud->header.frame_id, stamp, ros::Duration(0.05), &tfError)) {
        if (!pcl_ros::transformPointCloud(odom_child_frame_id, *pcl_cloud, *cloud, this->tf_buffer)) {
          ROS_ERROR("point cloud cannot be transformed into target frame!!");
          return;
        }
      } else {
        ROS_ERROR("cannot be transformed");
        ROS_ERROR("cloud->header.frame_id.is empty, shoule be %s . pcl_cloud->header.frame_id : %s", odom_child_frame_id.c_str(), pcl_cloud->header.frame_id.c_str());
        return;
      }
    } catch (tf::TransformException& ex) {
      ROS_WARN("%s", ex.what());
      NODELET_ERROR(tfError.c_str());
      return;
    }

    // 手动 从livox 转到body 坐标系下
    // cloud->points.clear();
    // cloud->points.resize( pcl_cloud->points.size() );

    // cloud->header.frame_id = odom_child_frame_id;
    // cloud->header.stamp = pcl_cloud->header.stamp;

    // static const Eigen::Vector3d Lidar_offset_to_IMU(0.05512, 0.02226, 0.0297); // Horizon

    // for(int i=0; i < pcl_cloud->points.size() ; i++)
    // {
    //   cloud->points[i].x = pcl_cloud->points[i].x + Lidar_offset_to_IMU[0];
    //   cloud->points[i].y = pcl_cloud->points[i].y + Lidar_offset_to_IMU[1];
    //   cloud->points[i].z = pcl_cloud->points[i].z + Lidar_offset_to_IMU[2];
    // }

    if (cloud->header.frame_id.empty()) {
      ROS_ERROR("cloud->header.frame_id.is empty, shoule be %s . ", odom_child_frame_id.c_str());
      return;
    }

    // std::cout << __FILE__ << ":" << __LINE__ << " cloud pc size is: " << cloud->points.size() << std::endl;
    auto filtered = downsample(cloud);

    // 点云已经是提取的 surf 点，不滤波
    // auto filtered = cloud;
    // filtered->header = cloud->header;
    // filtered->header.frame_id = odom_child_frame_id;

    // std::cout << __FILE__ << ":" << __LINE__ << " filtered pc size is: " << filtered->points.size() << std::endl;

    if (filtered->header.frame_id.empty()) {
      ROS_ERROR("filtered->header.frame_id.is empty");
      return;
    }

    last_scan = filtered;

    if (relocalizing) {
      delta_estimater->add_frame(filtered);
    }

    Eigen::Matrix4f before = pose_estimator->matrix();

    // predict
    if (!use_imu) {
      pose_estimator->predict(stamp);
    } else {
      // std::cout << __FILE__ << ":" << __LINE__ << "  use imu " << std::endl;
      std::lock_guard<std::mutex> lock(imu_data_mutex);
      auto imu_iter = imu_data.begin();
      for (imu_iter; imu_iter != imu_data.end(); imu_iter++) {
        if (stamp < (*imu_iter)->header.stamp) {
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

    // ROS_INFO("pc time with now tf, TIME diff is: %f ", (stamp - last_correction_time).toSec() );

    static int cnts = 0;
    static int no_odom_cnts = 0;

    if (private_nh.param<bool>("enable_robot_odometry_prediction", false) && !last_correction_time.isZero()) {
      geometry_msgs::TransformStamped odom_delta;
      if (tf_buffer.canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, ros::Duration(0.05))) {
        odom_delta = tf_buffer.lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, ros::Duration(0.05));
        // ROS_INFO("stmap : pc time with now tf, TIME diff is: %f ", (stamp - odom_delta.header.stamp).toSec());
      } else if (tf_buffer.canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, ros::Time(0), robot_odom_frame_id, ros::Duration(0.05))) {
        odom_delta = tf_buffer.lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, ros::Time(0), robot_odom_frame_id, ros::Duration(0.05));
        // ROS_WARN("ros::Time(0) : pc time with now tf, TIME diff is: %f ", (stamp - odom_delta.header.stamp).toSec());
      }

      if (odom_delta.header.stamp.isZero()) {
        NODELET_ERROR_STREAM("---------- failed to look up transform between " << cloud->header.frame_id << " and " << robot_odom_frame_id << " canTransform failed cnts:" << cnts++);
        // 20221226 daiti
        Eigen::Isometry3d delta = Eigen::Isometry3d::Identity();
        pose_estimator->predict_odom(delta.cast<float>().matrix());
      } else {
        Eigen::Isometry3d delta = tf2::transformToEigen(odom_delta);
        pose_estimator->predict_odom(delta.cast<float>().matrix());
      }
    } 

    // correct
    auto aligned = pose_estimator->correct(stamp, filtered);

    if (aligned_pub.getNumSubscribers()) {
      aligned->header.frame_id = "map";
      aligned->header.stamp = cloud->header.stamp;
      aligned_pub.publish(aligned);
    }

    publish_scan_matching_status(points_msg->header, aligned);

    // publish_odometry(stamp, pose_estimator->matrix());

    // ROS_INFO("locate use time is %lf  s", ros::Time::now().toSec() - s_t.toSec());
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

    if (use_global_localization) {
      NODELET_INFO("set globalmap for global localization!");
      hdl_global_localization::SetGlobalMap srv;
      pcl::toROSMsg(*globalmap, srv.request.global_map);

      if (!set_global_map_service.call(srv)) {
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
    if (last_scan == nullptr) {
      NODELET_INFO_STREAM("no scan has been received");
      return false;
    }

    relocalizing = true;
    delta_estimater->reset();
    pcl::PointCloud<PointT>::ConstPtr scan = last_scan;

    hdl_global_localization::QueryGlobalLocalization srv;
    pcl::toROSMsg(*scan, srv.request.cloud);
    srv.request.max_num_candidates = 1;

    if (!query_global_localization_service.call(srv) || srv.response.poses.empty()) {
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
    pose_estimator.reset(new hdl_localization::PoseEstimator(
      registration,
      ros::Time::now(),
      Eigen::Vector3f(p.x, p.y, p.z),
      Eigen::Quaternionf(q.w, q.x, q.y, q.z),
      private_nh.param<double>("cool_time_duration", 0.5)));
  }

  /**
   * @brief downsampling
   * @param cloud   input cloud
   * @return downsampled cloud
   */
  pcl::PointCloud<PointT>::ConstPtr downsample(const pcl::PointCloud<PointT>::ConstPtr& cloud) const {
    if (!downsample_filter) {
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
    if (tf_buffer.canTransform(robot_odom_frame_id, odom_child_frame_id, ros::Time(0))) {
      /*      */
      ros::Time s_t = ros::Time::now();
      geometry_msgs::TransformStamped map_wrt_frame = tf2::eigenToTransform(Eigen::Isometry3d(pose.inverse().cast<double>()));
      map_wrt_frame.header.stamp = stamp;
      map_wrt_frame.header.frame_id = odom_child_frame_id;
      map_wrt_frame.child_frame_id = "map";

      geometry_msgs::TransformStamped frame_wrt_odom = tf_buffer.lookupTransform(robot_odom_frame_id, odom_child_frame_id, ros::Time(0), ros::Duration(0.1));

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

      tf_broadcaster.sendTransform(odom_trans);

      // ROS_WARN("one use time is %lf  s",  ros::Time::now().toSec() - s_t.toSec()  ) ;
      // ROS_WARN(" two TM xyz is %lf, %lf, %lf ", odom_trans.transform.translation.x,  odom_trans.transform.translation.y, odom_trans.transform.translation.z );

      // s_t = ros::Time::now();

      /*
      // 后到前的变换据矩阵， T_mb : body frame 到 map frame
      geometry_msgs::TransformStamped T_mb = tf2::eigenToTransform(Eigen::Isometry3d(pose.cast<double>()));
      geometry_msgs::TransformStamped T_wb = tf_buffer.lookupTransform(robot_odom_frame_id, odom_child_frame_id , ros::Time(0), ros::Duration(0.1));
      Eigen::Isometry3d T_mb_eigen , T_wb_eigen, T_mw;
      tf::transformMsgToEigen (T_mb.transform , T_mb_eigen );
      tf::transformMsgToEigen (T_wb.transform , T_wb_eigen );
      T_mw =  T_mb_eigen  * T_wb_eigen.inverse();

      geometry_msgs::TransformStamped TM;
      tf::transformEigenToMsg (T_mw , TM.transform);
      TM.header.stamp = stamp;
      TM.header.frame_id = "map";
      TM.child_frame_id = robot_odom_frame_id;
      tf_broadcaster.sendTransform(TM);
      // ROS_WARN("two use time is %lf  s",  ros::Time::now().toSec() - s_t.toSec()  ) ;
      // ROS_WARN(" pub  TM : %s",robot_odom_frame_id.c_str()  );
      // ROS_WARN(" two TM xyz is %lf, %lf, %lf ", TM.transform.translation.x,  TM.transform.translation.y, TM.transform.translation.z );
      */
    } else {
      // if (private_nh.param<bool>("/hdl_localization_nodelet/enable_robot_odometry_prediction", false)) {
        // ROS_WARN("%s,  %s",robot_odom_frame_id.c_str() , odom_child_frame_id.c_str());
        geometry_msgs::TransformStamped odom_trans = tf2::eigenToTransform(Eigen::Isometry3d(pose.cast<double>()));
        odom_trans.header.stamp = stamp;
        odom_trans.header.frame_id = "map";
        odom_trans.child_frame_id = odom_child_frame_id;
        tf_broadcaster.sendTransform(odom_trans);
      // }
    }

    // publish the transform
    nav_msgs::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = "map";
    odom.twist.twist.linear.x = 0.0;
    odom.twist.twist.linear.y = 0.0;
    odom.twist.twist.angular.z = 0.0;

    tf::poseEigenToMsg(Eigen::Isometry3d(pose.cast<double>()), odom.pose.pose);
    odom.child_frame_id = odom_child_frame_id;
    pose_pub.publish(odom);

    // 把这个 定位成功的 位姿记录下来
    last_odom_pose.header = odom.header;
    last_odom_pose.pose.pose = odom.pose.pose;

    // double yaw = tf2::getYaw(status.relative_pose.rotation) * 180.0 /  3.14159;

    double roll, pitch, yaw;  //定义存储r\p\y的容器
    // tf::Matrix3x3(odom.pose.pose.orientation).getRPY(roll, pitch, yaw);//进行转换
    // yaw = yaw * 180.0 /  3.14159;
    yaw = tf::getYaw(odom.pose.pose.orientation) * 180.0 / 3.14159;

    std::ostringstream ossx;
    ossx << setiosflags(std::ios::fixed) << std::setprecision(2) << odom.pose.pose.position.x;
    std::string xs = ossx.str();

    std::ostringstream ossy;
    ossy << setiosflags(std::ios::fixed) << std::setprecision(2) << odom.pose.pose.position.y;
    std::string ys = ossy.str();

    std::ostringstream ossyaw;
    ossyaw << setiosflags(std::ios::fixed) << std::setprecision(1) << yaw;
    std::string yaws = ossyaw.str();

    std_msgs::String str_temp;

    str_temp.data = "x: " + xs + " y: " + ys + " yaw:" + yaws;
    // ROS_ERROR("str_temp.data  is : %s ", str_temp.data.c_str()  );
    string_pub.publish(str_temp);

    static std::ofstream ttttttt("/home/liunao/hdl/hdl_pose.txt", std::ios::out);
    static std::ofstream ofs;
    ofs.open("/home/liunao/hdl/hdl_pose.txt", std::ios::app);
    ofs << std::to_string(stamp.toSec()) << " " << std::to_string(odom.pose.pose.position.x) << " " << std::to_string(odom.pose.pose.position.y) << " "
        << std::to_string(odom.pose.pose.position.z) << " " << std::to_string(odom.pose.pose.orientation.x) << " " << std::to_string(odom.pose.pose.orientation.y) << " "
        << std::to_string(odom.pose.pose.orientation.z) << " " << std::to_string(odom.pose.pose.orientation.w) << std::endl;
    ofs.close();
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

    double max_correspondence_dist = 0.1;
    max_correspondence_dist = private_nh.param<double>("max_correspondence_dist", 0.1);

    int num_inliers = 0;
    std::vector<int> k_indices;
    std::vector<float> k_sq_dists;
    for (int i = 0; i < aligned->size(); i++) {
      const auto& pt = aligned->at(i);
      registration->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
      if (k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
        num_inliers++;
      }
    }
    status.inlier_fraction = static_cast<float>(num_inliers) / aligned->size();
    status.relative_pose = tf2::eigenToTransform(Eigen::Isometry3d(registration->getFinalTransformation().cast<double>())).transform;

    std_msgs::Float32 confidence;
    confidence.data = status.inlier_fraction * 100;
    confidence_pub.publish(confidence);

    status.prediction_labels.reserve(2);
    status.prediction_errors.reserve(2);

    std::vector<double> errors(6, 0.0);

    if (pose_estimator->wo_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = "without_pred";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->wo_prediction_error().get().cast<double>())).transform);
    }

    if (pose_estimator->imu_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = use_imu ? "imu" : "motion_model";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->imu_prediction_error().get().cast<double>())).transform);
    }

    if (pose_estimator->odom_prediction_error()) {
      status.prediction_labels.push_back(std_msgs::String());
      status.prediction_labels.back().data = "odom";
      status.prediction_errors.push_back(tf2::eigenToTransform(Eigen::Isometry3d(pose_estimator->odom_prediction_error().get().cast<double>())).transform);
    }

    status_pub.publish(status);

    // 匹配好的点<内点比例> 大于 该阈值认为 定位成功
    float LOC_SUCCESS_THRESHLOD = 0.90;
    LOC_SUCCESS_THRESHLOD = private_nh.param<double>("LOC_SUCCESS_THRESHLOD", LOC_SUCCESS_THRESHLOD);

    // ROS_WARN(" status.inlier_fraction %lf , time %f . aligned->size() %d ", status.inlier_fraction, header.stamp.toSec(), aligned->size());
    
    if (status.inlier_fraction > LOC_SUCCESS_THRESHLOD) 
    // if ( 1 ) 
    {
      // ROS_WARN(" publish_odometry ");
      publish_odometry(header.stamp, registration->getFinalTransformation());
    }
    else 
    {
      static int hdl_fail_cnts = 0;
      hdl_fail_cnts++;
      Eigen::Isometry3d PoseDelta = Eigen::Isometry3d::Identity();

      static int odom_interlp_ok_cnts = 0;

      // double last_time = last_odom_pose.header.stamp.toSec();

      geometry_msgs::TransformStamped odom_delta;

      // 时间戳调转一下
      if (tf_buffer.canTransform(odom_child_frame_id, header.stamp , odom_child_frame_id, last_odom_pose.header.stamp , robot_odom_frame_id, ros::Duration(0.05))) {
        odom_delta = tf_buffer.lookupTransform(odom_child_frame_id, header.stamp, odom_child_frame_id, last_odom_pose.header.stamp  , robot_odom_frame_id, ros::Duration(0.05));
        PoseDelta = tf2::transformToEigen( odom_delta ).inverse() ;
        ROS_ERROR_STREAM("pose delta: " << std::endl << PoseDelta.matrix() << std::endl);
      }
      else
      {
        // odom_delta.transform.translation.x = last_odom_pose.twist.twist.linear.x * 0.2 ;
        // odom_delta.transform.translation.y = last_odom_pose.twist.twist.linear.y * 0.2 ;
        // odom_delta.transform.translation.z = last_odom_pose.twist.twist.linear.z * 0.2 ;
        // PoseDelta = tf2::transformToEigen(odom_delta);
        ROS_ERROR(" can not interlp-------------- ");
        return;
      }

      ++odom_interlp_ok_cnts;

      ROS_INFO(" hdl_fail_cnts %ld, odom_interlp_ok_cnts: %ld. ,precent: %lf", hdl_fail_cnts, odom_interlp_ok_cnts, (1.0*odom_interlp_ok_cnts) / (1.0*hdl_fail_cnts) );

      nav_msgs::Odometry odom;
      odom.header.stamp = header.stamp;
      odom.header.frame_id = "map";
      // odom.twist.twist.linear.x = 0.0;
      // odom.twist.twist.linear.y = 0.0;
      // odom.twist.twist.angular.z = 0.0;

      Eigen::Isometry3d last_pose_eigen;
      tf2::fromMsg(last_odom_pose.pose.pose, last_pose_eigen);

      Eigen::Isometry3d now_pose_eigen;
      // now_pose_eigen = PoseDelta * last_pose_eigen;
      // publish_odometry(header.stamp, now_pose_eigen.matrix().cast<float>());
      now_pose_eigen = last_pose_eigen * PoseDelta;
      publish_odometry(header.stamp, now_pose_eigen.matrix().cast<float>());

      //TODO 里程计补偿的 位姿上，匹配度有多少  大于hdl的匹配度，再决定用哪个位姿 ？
      
      // // Executing the transformation
      // pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud_o (new pcl::PointCloud<pcl::PointXYZI> ());
      // pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZI> ());

      // pcl::transformPointCloud (*aligned, *transformed_cloud_o, registration->getFinalTransformation().inverse() );
      // pcl::transformPointCloud (*transformed_cloud_o, *transformed_cloud, now_pose_eigen.matrix());      
      
      // sensor_msgs::PointCloud2 output_msg;
      // pcl::toROSMsg(*transformed_cloud, output_msg);
      // output_msg.header.frame_id="map";
      // output_msg.header.stamp = header.stamp;
      // aligned_pub.publish(output_msg);

      // num_inliers = 0;
      // for (int i = 0; i < transformed_cloud->size(); i++) {
      // const auto& pt = transformed_cloud->at(i);
      // registration->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
      // if (k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
      //   num_inliers++;
      // }
      // }
      // double now_inlier_fraction = static_cast<float>(num_inliers) / transformed_cloud->size();
      // ROS_INFO("now_inlier_fraction %f ", now_inlier_fraction);
/*
      tf::poseEigenToMsg(Eigen::Isometry3d(now_pose_eigen.cast<double>()), odom.pose.pose);
      odom.child_frame_id = odom_child_frame_id;

      // record now odom data
      last_odom_pose.header = odom.header;
      last_odom_pose.pose.pose = odom.pose.pose;
      // last_odom_pose.twist.twist = odom.twist.twist;
      
      pose_pub.publish(odom);

*/
      // ROS_ERROR_STREAM("pose delta: " << PoseDelta.matrix() << std::endl);
    }
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
  ros::Subscriber odom_sub;

  ros::Publisher pose_pub;
  ros::Publisher aligned_pub;
  ros::Publisher status_pub;
  ros::Publisher confidence_pub;
  ros::Publisher string_pub;

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

  bool no_odom_flag;
  ros::Time time_1, time_30;

  // odom input buffer
  std::mutex odom_data_mutex;
  std::vector<nav_msgs::OdometryConstPtr> odom_data;
  // geometry_msgs::PoseStamped last_odom_pose;
  nav_msgs::Odometry last_odom_pose;

};
}  // namespace hdl_localization

PLUGINLIB_EXPORT_CLASS(hdl_localization::HdlLocalizationNodelet, nodelet::Nodelet)
