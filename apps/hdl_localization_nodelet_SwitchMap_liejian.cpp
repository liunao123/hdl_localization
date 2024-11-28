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

#include <std_msgs/Float32.h>
#include <std_msgs/String.h>
#include <std_srvs/Empty.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <yaml-cpp/yaml.h>

#include <tf/transform_broadcaster.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/common/transforms.h>
#include <pclomp/ndt_omp.h>
#include <fast_gicp/ndt/ndt_cuda.hpp>

#include <hdl_localization/pose_estimator.hpp>
#include <hdl_localization/delta_estimater.hpp>

#include <hdl_localization/ScanMatchingStatus.h>
#include <hdl_global_localization/SetGlobalMap.h>
#include <hdl_global_localization/QueryGlobalLocalization.h>

#include <base_controller/csgPoseStampedMatchValue.h>
#include <base_controller/CSGMapInfo.h>

#include "zlog.h"

namespace hdl_localization {

class HdlLocalizationNodelet : public nodelet::Nodelet {
public:
  using PointT = pcl::PointXYZI;

  HdlLocalizationNodelet() : tf_buffer(), tf_listener(tf_buffer) {}
  virtual ~HdlLocalizationNodelet() { dzlog_info(" Note : EXIT HDL localization node .^.^.^."); }

  void onInit() override {
    initZlog();
    dzlog_info("strat HDL localization node .......");
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
    switchMapSts_sub = nh.subscribe("/switchMap", 2, &HdlLocalizationNodelet::switchMapStatusCallBack, this);

    pose_pub = nh.advertise<nav_msgs::Odometry>("odom", 1, false);
    aligned_pub = nh.advertise<sensor_msgs::PointCloud2>("aligned_points", 1, false);
    status_pub = nh.advertise<ScanMatchingStatus>("/status", 5, false);
    confidence_pub = nh.advertise<std_msgs::Float32>("/confidece", 5, false);
    string_pub = nh.advertise<std_msgs::String>("/XY_yaw", 5, false);
    map_name_pub = nh.advertise<std_msgs::String>("/map_request/pcd", 1, false);
    current_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/current_pose", 5);
    keda_current_pub = nh.advertise<base_controller::csgPoseStampedMatchValue>("/keda_current_pose", 5);

    pub_map_info_status_ = nh.advertise<base_controller::CSGMapInfo>("/switchMap_Status", 10, false);
    map_info_pub_timer_ = nh.createTimer(ros::Duration(1), &HdlLocalizationNodelet::MapInfoPubTimer, this);

    iStatus = 0;
    dzlog_info("load_map_from_yaml");
    load_map_from_yaml();

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

    dzlog_info("unknown registration method: %s ", reg_method.c_str());
    return nullptr;
  }

  void switchMapStatusCallBack(const base_controller::CSGMapInfo& mapInfo) {
    iRegionID = mapInfo.iRegionID;
    iMapID = mapInfo.iMapID;
    iNavType = mapInfo.iNavType;

    // write to file
    std::fstream fs;
    std::string dataPath = "/home/map/mapinfo.yaml";
    fs.open(dataPath.c_str(), std::fstream::out);
    if (fs.is_open()) {
      fs << iRegionID << std::endl;
      fs << iMapID << std::endl;
      fs << iNavType << std::endl;
      fs.close();
    } else {
      dzlog_error("###### saveMapInfo() cannot open mapinfo.yaml !!!");
    }
    dzlog_info("@@@@@@ saveMapInfo() write mapInfo OK!!!");

    if (mapInfo.iNavType != 5) return;

    load_global_map();
  }

  void MapInfoPubTimer(const ros::TimerEvent& time) {
    if (iNavType != 5) return;

    if ((pub_map_info_status_.getNumSubscribers() > 0)) {
      base_controller::CSGMapInfo mapInfo;
      mapInfo.iRegionID = iRegionID;
      mapInfo.iMapID = iMapID;
      mapInfo.iStatus = iStatus;
      mapInfo.iNavType = iNavType;  // test temp
      pub_map_info_status_.publish(mapInfo);
    }
  }

  int initZlog() {
    if (-1 == access("/home/roslog", F_OK)) {
      mkdir("/home/roslog", 0777);
    }
    if (dzlog_init("/home/config/zlog.conf", "hdl_cat") != 0) {
      printf("@@@ init zlog failed\n");
      return -1;
    }
    return 0;
  }

  void load_map_from_yaml() {
    iRegionID = 0;
    iMapID = 0;
    iNavType = -1;

    std::fstream fs;
    std::string mapFile = "/home/map/mapinfo.yaml";
    fs.open(mapFile.c_str(), std::fstream::in);
    if (fs.is_open()) {
      char line[100];
      fs.getline(line, 100);
      iRegionID = std::atoi(line);
      fs.getline(line, 100);
      iMapID = std::atoi(line);
      fs.getline(line, 100);
      iNavType = std::atoi(line);
      fs.close();
      dzlog_info("@@@@@@ loadMapInfo() iRegionID = %d iMapID = %d,iNavType = %d", iRegionID, iMapID, iNavType);
    } else {
      dzlog_error("###### loadMapInfo() cannot read mapInfo data !");
      return;
    }

    dzlog_info("load_global_map: ");
    if (iNavType != 5) {
      dzlog_info("iNavType is NOT 5 return : ");
      return;
    }
    
    load_global_map();
    
  }

  void load_global_map() {
    std_msgs::String pn;
    pn.data = "/home/map/region" + std::to_string(iRegionID) + "-" + std::to_string(iMapID) + "/saveMap.pcd";
    iStatus = 1;
    ros::Duration(1).sleep();
    map_name_pub.publish(pn);
    dzlog_info("iRegionID is %d , iMapID is %d, iNavType is %d . map file %s .", iRegionID, iMapID, iNavType, pn.data.c_str());
    ros::Duration(2).sleep();
    iStatus = 2;

    // read init pose from file
    std::fstream fs;
    std::string initdatafile = "/home/map/init.yaml";
    fs.open(initdatafile.c_str(), std::fstream::in);
    if (fs.is_open()) {
      char line[100];
      fs.getline(line, 100);
      float poseX = std::atof(line);
      fs.getline(line, 100);
      float poseY = std::atof(line);
      fs.getline(line, 100);
      float poseYaw = std::atof(line);
      fs.close();

      if (std::isnan(poseX) || std::isinf(poseX)) {
        poseX = 0.0;
      }
      if (std::isnan(poseY) || std::isinf(poseY)) {
        poseY = 0.0;
      }
      if (std::isnan(poseYaw) || std::isinf(poseYaw)) {
        poseYaw = 0.0;
      }
      
      float mean_z = get_z_from_map_by_xy( poseX, poseY );
      dzlog_info("@@@@@@ loadLastestPose() from init.yaml. init_x = %f,y = %f,z<computed> = %f,  yaw = %f", poseX, poseY, mean_z , poseYaw);

      pose_estimator.reset(new hdl_localization::PoseEstimator(registration,
        Eigen::Vector3f(poseX, poseY, mean_z),
        Eigen::Quaternionf(cos(poseYaw / 2.0), 0, 0, sin(poseYaw / 2.0)),
        private_nh.param<double>("cool_time_duration", 0.5)));
    } else {
      dzlog_info("@@@@@@ loadLastestPose() cannot read init pose file ,try to get from rosparam !!!");
      // initialize pose estimator
      if (private_nh.param<bool>("specify_init_pose", true)) {
        NODELET_INFO("initialize pose estimator with specified parameters!!");
        dzlog_info("@@@@@@ load init Pose() from rosparam , init_pos_x = %f, init_pos_y = %f, init_pos_z = %f ...... " , private_nh.param<double>("init_pos_x", 0.0) , private_nh.param<double>("init_pos_y", 0.0) , private_nh.param<double>("init_pos_z", 0.0));
        pose_estimator.reset(new hdl_localization::PoseEstimator(registration,
          Eigen::Vector3f(private_nh.param<double>("init_pos_x", 0.0), private_nh.param<double>("init_pos_y", 0.0), private_nh.param<double>("init_pos_z", 0.0)),
          Eigen::Quaternionf(
            private_nh.param<double>("init_ori_w", 1.0),
            private_nh.param<double>("init_ori_x", 0.0),
            private_nh.param<double>("init_ori_y", 0.0),
            private_nh.param<double>("init_ori_z", 0.0)),
          private_nh.param<double>("cool_time_duration", 0.5)));
      }
    }
  
  }

  void initialize_params() {
    // intialize scan matching method
    double downsample_resolution = private_nh.param<double>("downsample_resolution", 0.1);
    boost::shared_ptr<pcl::VoxelGrid<PointT>> voxelgrid(new pcl::VoxelGrid<PointT>());
    voxelgrid->setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    downsample_filter = voxelgrid;

    rangeMinX = private_nh.param<double>("rangeMinX", -250);
    rangeMaxX = private_nh.param<double>("rangeMaxX", 250);
    rangeMinY = private_nh.param<double>("rangeMinY", -250);
    rangeMaxY = private_nh.param<double>("rangeMaxY", 250);
    rangeMinZ = private_nh.param<double>("rangeMinZ", -250);
    rangeMaxZ = private_nh.param<double>("rangeMaxZ", 250);

    dzlog_info("create registration method for localization");
    registration = create_registration();

    // global localization
    dzlog_info("create registration method for fallback during relocalization");
    relocalizing = false;
    relocate = true;

    wait_tf_duration =  ros::Duration(0.05);

    delta_estimater.reset(new DeltaEstimater(create_registration()));

    // initialize pose estimator
    if(private_nh.param<bool>("specify_init_pose", true)) {
      NODELET_INFO("initialize pose estimator with specified parameters!!");
      pose_estimator.reset(new hdl_localization::PoseEstimator(registration,
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
    static int cnt_lidar = 0;
    if (iNavType != 5 || iStatus != 2)  // 导航类型不对 or  地图没有加载好
    {
      if (cnt_lidar++ > 20) {
        cnt_lidar = 0;
        dzlog_info("iNavType != 5 , DO NOT USE HDL localization type , return ......");
      }
      return;
    }

    if (!globalmap) {
      dzlog_info("globalmap has not been received!!");
      return;
    }

    const auto& stamp = points_msg->header.stamp;
    pcl::PointCloud<PointT>::Ptr pcl_cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *pcl_cloud);

    // 激光雷达，去除 nan 点
    // dzlog_info("pl_orig->size()is %d",  pcl_cloud->points.size());
    pcl_cloud->is_dense = false; // 万集的雷达必须加这一句
    std::vector<int> save_index;
    pcl::removeNaNFromPointCloud(*pcl_cloud, *pcl_cloud, save_index);
    // dzlog_info("pl_orig->size()is %d",  pcl_cloud->points.size());

    if (pcl_cloud->empty()) {
      dzlog_info("cloud is empty!!");
      return;
    }

    // transform pointcloud into odom_child_frame_id
    std::string tfError;
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
	
    /*    *cloud =  *pcl_cloud;  */
    std::string points_frame = points_msg->header.frame_id;

    static Eigen::Matrix4f TF_temp = Eigen::Matrix4f::Identity();
    static bool get_tf_ok = false;

    // if ( !get_tf_ok )
    // {
    //   // tf 是静态的，等待时间长一点也无所谓
    //   try {
    //     if (tf_buffer.canTransform(odom_child_frame_id, points_frame, ros::Time(0), ros::Duration(0.5), &tfError)) {
    //       geometry_msgs::TransformStamped TF_lidar2odom_child_frame = tf_buffer.lookupTransform(odom_child_frame_id, points_frame,  ros::Time(0) , ros::Duration(0.5));
    //       TF_temp = tf2::transformToEigen(TF_lidar2odom_child_frame).cast<float>().matrix();
		//       //std::cout <<  TF_temp  << std::endl;
    //       get_tf_ok = true;
    //       dzlog_info(" get transformed success . between %s . and : %s ", odom_child_frame_id.c_str(), points_frame.c_str() );
    //     } else {
    //       dzlog_info("cannot be transformed . odom_child_frame_id is %s . points_frame : %s . may use Identity()", odom_child_frame_id.c_str(), points_frame.c_str());
    //     }
    //   } catch (tf::TransformException& ex) {
    //     dzlog_info("%s", ex.what());
    //     dzlog_info(tfError.c_str());
    //     // return;
    //   }
    // }

    // exe transform points
		pcl::transformPointCloud(*pcl_cloud, *cloud, TF_temp);
		// cloud->header.frame_id = odom_child_frame_id;
    // dzlog_info("pl_orig->size()is %d",  cloud->points.size());

    auto filtered = downsample(cloud);

    last_scan = filtered;

    if(relocalizing) {
      delta_estimater->add_frame(filtered);
    }

    std::lock_guard<std::mutex> estimator_lock(pose_estimator_mutex);
    if(!pose_estimator) {
      NODELET_ERROR("waiting for initial pose input!!");
      return;
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
        // if ( std::fabs(acc.x) > 10 || std::fabs(acc.y) > 10 || std::fabs(acc.z-10) > 10 || std::fabs(gyro.x) > 1 || std::fabs(gyro.y) > 1 || std::fabs(gyro.z) > 1 )
        // if ( std::fabs(gyro.x) > 1 || std::fabs(gyro.y) > 1 || std::fabs(gyro.z) > 1 )
        // {
        //   continue;
        // }
        // pose_estimator->predict((*imu_iter)->header.stamp, acc_sign * Eigen::Vector3f(acc.x, acc.y, acc.z) , gyro_sign * Eigen::Vector3f(gyro.x, gyro.y, gyro.z));
        pose_estimator->predict((*imu_iter)->header.stamp, acc_sign * Eigen::Vector3f(acc.x, acc.y, acc.z) * 9.80665 , gyro_sign * Eigen::Vector3f(gyro.x, gyro.y, gyro.z));
      }
      imu_data.erase(imu_data.begin(), imu_iter);
    }

    // odometry-based prediction
    ros::Time last_correction_time = pose_estimator->last_correction_time();

    // dzlog_info("pc time with now tf, TIME diff is: %f ", (stamp - last_correction_time).toSec() );

    if (private_nh.param<bool>("enable_robot_odometry_prediction", false) && !last_correction_time.isZero()) {
      geometry_msgs::TransformStamped odom_delta;
      if (tf_buffer.canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, wait_tf_duration)) {
        odom_delta = tf_buffer.lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, stamp, robot_odom_frame_id, wait_tf_duration);
        // dzlog_info("stmap : pc time with now tf, TIME diff is: %f ", (stamp - odom_delta.header.stamp).toSec());
      } else if (tf_buffer.canTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, ros::Time(0), robot_odom_frame_id, wait_tf_duration)) {
        odom_delta = tf_buffer.lookupTransform(odom_child_frame_id, last_correction_time, odom_child_frame_id, ros::Time(0), robot_odom_frame_id, wait_tf_duration);
        // dzlog_info("ros::Time(0) : pc time with now tf, TIME diff is: %f ", (stamp - odom_delta.header.stamp).toSec());
      }

      if (odom_delta.header.stamp.isZero()) {
        // dzlog_info(" odom_delta is zero ..... use Identity() ");
        // 20221226 daiti
        // Eigen::Isometry3d delta = Eigen::Isometry3d::Identity();
        // pose_estimator->predict_odom(delta.cast<float>().matrix());
      } else {
        Eigen::Isometry3d delta = tf2::transformToEigen(odom_delta);
        pose_estimator->predict_odom(delta.cast<float>().matrix());
      }
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZI>());
    // ROS_INFO(" %ld points ", filtered->points.size() );
    // ! todo 20231010 直接使用里程计补偿定位
    // ? done 20231030 利用 robot_pose_ekf 融合后的里程计补偿完毕 test DONE
    // ! 
    int low_points_thresh = private_nh.param<int>("low_points_thresh", 10);
    // ！重定位的时候，不进行里程计补偿
    // if ( (filtered->points.size() < low_points_thresh)  &&  ("livox" == points_frame) && ( !relocate )   )
    if ( 0  )
    {
      pcl::PointCloud<pcl::PointXYZI>::Ptr tt(new pcl::PointCloud<pcl::PointXYZI>());
      aligned = pose_estimator->correct(stamp, tt);

      // wait_tf_duration
      Eigen::Isometry3d PoseDelta = Eigen::Isometry3d::Identity();
      geometry_msgs::TransformStamped odom_delta;

      double wait_time = private_nh.param<double>("wait_time", 0.02);
      if (tf_buffer.canTransform(odom_child_frame_id,                last_odom_pose.header.stamp , odom_child_frame_id, stamp, robot_odom_frame_id, ros::Duration(wait_time))) {
        odom_delta = tf_buffer.lookupTransform(odom_child_frame_id,  last_odom_pose.header.stamp , odom_child_frame_id, stamp, robot_odom_frame_id, ros::Duration(wait_time));
        PoseDelta = tf2::transformToEigen(odom_delta);
        // std::cout <<  std::endl << PoseDelta.matrix() << std::endl << std::endl;
        // dzlog_info("tf ok ");
      }
      else
      {
        // wait_time = 0.1;
        dzlog_info("ros::Time(0) %f ,lidar time %f ", odom_delta.header.stamp.toSec(), stamp.toSec() );
        try{
            odom_delta = tf_buffer.lookupTransform(odom_child_frame_id, last_odom_pose.header.stamp , odom_child_frame_id, ros::Time(0) ,  robot_odom_frame_id  );
            PoseDelta = tf2::transformToEigen(odom_delta);
          } catch (tf2::TransformException &ex) {
            dzlog_info("WARN tf eeeeerror. Could NOT transform: %s",  ex.what());
            dzlog_info("WARN odom is not get . try hdl locate .");
            aligned = pose_estimator->correct(stamp, filtered);
            publish_scan_matching_status(points_msg->header, aligned);
            return;
           }
        // dzlog_info(", between %s and %s . wait time %f ", odom_child_frame_id.c_str(), robot_odom_frame_id.c_str() , wait_time );
      }

      // ! 里程计补偿后，方向不对，导致匹配度很低 20231020
      // * ros odom与eigen格式来回转换导致的精度下降 全部使用eigen即可 20231030
      Eigen::Isometry3d now_pose_eigen = last_pose_eigen * PoseDelta;
      // std::cout <<  std::endl << last_pose_eigen.matrix() << std::endl << std::endl;
      // std::cout <<  std::endl << now_pose_eigen.matrix() << std::endl << std::endl;
      
      // 把当前点云 按补偿后的 位姿 投影到地图中，然后查找，得到匹配度
      pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>());
      pcl::transformPointCloud(*last_scan, *transformed_cloud, now_pose_eigen.matrix());
      // * 

      double max_correspondence_dist = private_nh.param<double>("max_correspondence_dist", 0.1);
      int num_inliers = 0;
      std::vector<int> k_indices;
      std::vector<float> k_sq_dists;
      for (int i = 0; i < transformed_cloud->size(); i++) {
        const auto& pt = transformed_cloud->at(i);
        registration->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
        if (k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist * 3 ) { // ! 近距离测距精度差 扩大范围提高匹配度 
          num_inliers++;
        }
      }
      double compensate_inlier_fraction = static_cast<float>(num_inliers) / transformed_cloud->size();
      dzlog_info("livox points is %d . less %d .use leg_odom to get pose . compensate_confidence_with_leg_odom %lf ", filtered->points.size() , low_points_thresh , compensate_inlier_fraction);

      publish_odometry( stamp, now_pose_eigen.matrix().cast<float>(), compensate_inlier_fraction);  // 发布用里程计补偿的位姿
      *aligned = *transformed_cloud;
      use_odom_flag = true;
      // ! end use leg_odom
    }
    else
    {
      use_odom_flag = false;
      // correct
      aligned = pose_estimator->correct(stamp, filtered);
    }

    if ( !use_odom_flag )
    {
      publish_scan_matching_status(points_msg->header, aligned);
    }

    if (aligned_pub.getNumSubscribers()) {
      pcl::RandomSample< pcl::PointXYZI > randFilter;
      randFilter.setInputCloud(aligned);
      randFilter.setSample(1200);
      randFilter.filter(*aligned);

      aligned->header.frame_id = "map";
      aligned->header.stamp = cloud->header.stamp;
      aligned_pub.publish(aligned);
    }

  }

  /**
   * @brief callback for globalmap input
   * @param points_msg
   */
  void globalmap_callback(const sensor_msgs::PointCloud2ConstPtr& points_msg) {
    dzlog_info("globalmap received!");
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(*points_msg, *cloud);
    globalmap = cloud;

    registration->setInputTarget(globalmap);
    dzlog_info("Get a new global map .  DONE ...... ");

    // if (use_global_localization) {
    //   NODELET_INFO("set globalmap for global localization!");
    //   hdl_global_localization::SetGlobalMap srv;
    //   pcl::toROSMsg(*globalmap, srv.request.global_map);

    //   if (!set_global_map_service.call(srv)) {
    //     NODELET_INFO("failed to set global map");
    //   } else {
    //     NODELET_INFO("done");
    //   }
    // }

  }

  /**
   * @brief perform global localization to relocalize the sensor position
   * @param
   */
  bool relocalize(std_srvs::EmptyRequest& req, std_srvs::EmptyResponse& res) {
    // dzlog_info("Try global localization.");
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
      pose.translation(),
      Eigen::Quaternionf(pose.linear()),
      private_nh.param<double>("cool_time_duration", 0.5)));

    relocalizing = false;

    return true;
  }

  /**
   * @brief get Z from map , by given XY value
   * @param px py
   * @return  average Z within 3m
   */
  float get_z_from_map_by_xy(const float px, const float py)
  {
    int cnts_try = 0;
    while ( !globalmap && cnts_try < 100 ) {
      dzlog_info(" globalmap has not been received!!");
      ros::Duration(1,0).sleep();
      cnts_try++;
    }

    if ( cnts_try > 90 )
    {
      dzlog_info("cnts_try > 90,  WATI 100s the global map is NOT be load , return ... _-_ ...");
      return 0.0;
    }
    float acc_z = 0;
    float acc_z_2 = 0;
    float mean_z = 0;
    int num_inlier = 0;
    std::vector<int> k_indices;

    dzlog_info(" start to find z from initialpose ." );
    for (int i = 0; i < globalmap->size(); i += 10) {
      const auto& pt = globalmap->at(i);
      if( std::fabs(px - pt.x) < 2.0 && std::fabs(py - pt.y) < 2.0  ) // 更普遍 3m以内的点取均值
      {
        acc_z  += globalmap->points[ i ].z;
        k_indices.push_back( i );
      }
    }

    if ( k_indices.size() > 0 ) 
    {
      dzlog_info(" mean z first: is :  %f .<cnts is %ld > ", acc_z / k_indices.size(), k_indices.size());
      for (int i = 0; i < k_indices.size(); i++) {
        if (globalmap->points[k_indices[i]].z <= acc_z / k_indices.size()) {
          acc_z_2 += globalmap->points[k_indices[i]].z;
          num_inlier++;
        }
      }
      mean_z = acc_z_2 / num_inlier;
      dzlog_info(" mean z second is :  %f .<cnts is %d > ", mean_z , num_inlier);
    }
    else
    {
      dzlog_error(" This initialpose is INVAILD , relocate again please ..... " );
    }
    dzlog_info(" mean z is :  %f .<cnts is %d > ", mean_z , num_inlier);
    dzlog_info("@@@@@@ set /initialpose. x = %f,y = %f, z = %f", px, py, mean_z);

    // end of get z values
    // mean_z = -1.8;
    return mean_z + 0.0; // add 0.2 mean robot is higher then the ground 0.2m.
  }


  /**
   * @brief callback for initial pose input ("2D Pose Estimate" on rviz)
   * @param pose_msg
   */
  void initialpose_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose_msg) {
    dzlog_info("NOTE : initial pose received!!");
    if (iNavType != 5) {
      dzlog_info("iNavType != 5 , DO NOT USE HDL localization type , NOT execute relocate cmd return ;");
      return;
    }

    const auto& p = pose_msg->pose.pose.position;
    const auto& q = pose_msg->pose.pose.orientation;

    tf::Quaternion quat;
    tf::quaternionMsgToTF(q, quat);
    double roll, pitch, yaw;//定义存储roll,pitch and yaw的容器
    tf::Matrix3x3(quat).getRPY(roll, pitch, yaw); //进行转换
    relocate = true;
  	dzlog_info(" keep relocate =  true for 0.1s ");

    // 手动指定 机器人的高  
    // 当重定位的时候 自动计算出来地图上 该位置附近一定范围内地面点的高度 取均值来作为机器人的高度值
    if (!globalmap) {
      ROS_WARN("globalmap has not been received!!");
    }

    float mean_z = get_z_from_map_by_xy( p.x, p.y );
    dzlog_info(" initial x. y. z. yaw<degree> : %lf, %lf, %f, %lf", p.x, p.y, mean_z, yaw * 180.0/3.14);
    last_pose_eigen.pretranslate ( Eigen::Vector3d(p.x, p.y, mean_z) );
    last_pose_eigen.rotate ( Eigen::Quaterniond(q.w, q.x, q.y, q.z).matrix() );
    
    dzlog_info(" lock(pose_estimator_mutex) ");
    std::lock_guard<std::mutex> lock(pose_estimator_mutex);
    pose_estimator.reset(new hdl_localization::PoseEstimator(
      registration,

      Eigen::Vector3f(p.x, p.y, mean_z),
      Eigen::Quaternionf(q.w, q.x, q.y, q.z),
      private_nh.param<double>("cool_time_duration", 0.5)));

  	dzlog_info(" lock(pose_estimator_mutex) DONE ");

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

    // ROS_WARN(" pc size() %d ", filtered->size());
    // 保留机器人上下 2m 以内的点，其余的点可能会打在 列车上(如果有)，无法使用。
    static pcl::CropBox<PointT> cropBoxFilter_temp(false);
  
    cropBoxFilter_temp.setMin(Eigen::Vector4f(rangeMinX, rangeMinY, rangeMinZ, 1.0f));
    cropBoxFilter_temp.setMax(Eigen::Vector4f(rangeMaxX, rangeMaxY, rangeMaxZ, 1.0f));
    cropBoxFilter_temp.setNegative(false);
    cropBoxFilter_temp.setInputCloud(cloud);
    cropBoxFilter_temp.filter(*filtered);
    
    // ROS_WARN(" pc size() %d ", filtered->size());
    // 去除距离很近的点
    float range = 1.0;
    cropBoxFilter_temp.setMin(Eigen::Vector4f(-range, -range, -range, 1.0f));
    cropBoxFilter_temp.setMax(Eigen::Vector4f(range, range, range, 1.0f));
    cropBoxFilter_temp.setNegative(true);
    cropBoxFilter_temp.setInputCloud(filtered);
    cropBoxFilter_temp.filter(*filtered);

    downsample_filter->setInputCloud(filtered);
    downsample_filter->filter(*filtered);
    filtered->header = cloud->header;

    return filtered;
  }

  /**
   * @brief publish odometry
   * @param stamp  timestamp
   * @param pose   odometry pose to be published
   */
  void publish_odometry(const ros::Time& stamp, const Eigen::Matrix4f& pose, const double MatchValue) {
    if ( std::isnan(pose(0,3)) || std::isnan(pose(1,3)) || std::isnan(pose(2,3)))
    {
      dzlog_info(" ERROR NAN POSE ...-_-... return ") ;
      return;
    }
    // broadcast the transform over tf
    if(tf_buffer.canTransform(robot_odom_frame_id, odom_child_frame_id, ros::Time(0))) {
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

      // dzlog_info(" two TM xyz is %lf, %lf, %lf ", odom_trans.transform.translation.x,  odom_trans.transform.translation.y, odom_trans.transform.translation.z );

      /*
      // 后到前的变换据矩阵， T_mb : body frame 到 map frame
      geometry_msgs::TransformStamped T_mb = tf2::eigenToTransform(Eigen::Isometry3d(pose.cast<double>()));
      geometry_msgs::TransformStamped T_wb = tf_buffer.lookupTransform(robot_odom_frame_id, odom_child_frame_id , ros::Time(0), wait_tf_duration);
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
      // dzlog_info(" pub  TM : %s",robot_odom_frame_id.c_str()  );
      // dzlog_info(" two TM xyz is %lf, %lf, %lf ", TM.transform.translation.x,  TM.transform.translation.y, TM.transform.translation.z );
      */
    } else {
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
    odom.twist.twist.linear.x = ( pose(0,3) - last_pose_eigen(0,3) ) / ( stamp - last_odom_pose.header.stamp ).toSec();
    odom.twist.twist.linear.y = ( pose(1,3) - last_pose_eigen(1,3) ) / ( stamp - last_odom_pose.header.stamp ).toSec();
    odom.twist.twist.linear.z = ( pose(2,3) - last_pose_eigen(2,3) ) / ( stamp - last_odom_pose.header.stamp ).toSec();
    Eigen::Vector3d last_eulerAngle = last_pose_eigen.rotation().eulerAngles(2,1,0);
    Eigen::Vector3d now_eulerAngle = Eigen::Isometry3d(pose.cast<double>()).rotation().eulerAngles(2,1,0);
    Eigen::Vector3d eulerAngle_velocity = ( now_eulerAngle - last_eulerAngle ) / ( stamp - last_odom_pose.header.stamp ).toSec() ;
    odom.twist.twist.angular.z = eulerAngle_velocity(0);
    odom.twist.twist.angular.y = eulerAngle_velocity(1);
    odom.twist.twist.angular.x = eulerAngle_velocity(2);

    tf::poseEigenToMsg(Eigen::Isometry3d(pose.cast<double>()), odom.pose.pose);
    odom.child_frame_id = odom_child_frame_id;
    pose_pub.publish(odom);

    // pub current pose and pose with match value
    geometry_msgs::PoseStamped current_pose;
    current_pose.header = odom.header;
    current_pose.pose = odom.pose.pose;
    current_pose_pub.publish(current_pose);

    base_controller::csgPoseStampedMatchValue cpm;
    cpm.header = odom.header;
    cpm.pose = current_pose.pose;
    cpm.match_value = MatchValue;

    // 暂定
    cpm.iRegionID = iRegionID;
    cpm.iMapID = iMapID;
    keda_current_pub.publish(cpm);

    // 把这个 定位成功的 位姿记录下来
    last_pose_eigen = Eigen::Isometry3d(pose.cast<double>());
    
    // 把这个时间戳记录下来
    last_odom_pose.header = odom.header;

    static std_msgs::Float32 confidence;
    confidence.data = MatchValue * 100;
    confidence_pub.publish(confidence);

    double roll, pitch, yaw;  // 定义存储r\p\y的容器
    tf::Quaternion quaternion;
    tf::quaternionMsgToTF(odom.pose.pose.orientation, quaternion);
    tf::Matrix3x3(quaternion).getRPY(roll, pitch, yaw);//进行转换
    roll = roll * 180.0 /  3.14159;
    pitch = pitch * 180.0 /  3.14159;
    yaw = yaw * 180.0 /  3.14159;

    // yaw = tf::getYaw(odom.pose.pose.orientation) * 180.0 / 3.14159;

    std::ostringstream ossx;
    ossx << setiosflags(std::ios::fixed) << std::setprecision(3) << odom.pose.pose.position.x;
    std::string xs = ossx.str();

    std::ostringstream ossy;
    ossy << setiosflags(std::ios::fixed) << std::setprecision(3) << odom.pose.pose.position.y;
    std::string ys = ossy.str();

    std::ostringstream ossz;
    ossz << setiosflags(std::ios::fixed) << std::setprecision(3) << odom.pose.pose.position.z;
    std::string zs = ossz.str();

    std::ostringstream ossroll;
    ossroll << setiosflags(std::ios::fixed) << std::setprecision(1) << roll;

    std::ostringstream osspitch;
    osspitch << setiosflags(std::ios::fixed) << std::setprecision(1) << pitch;
    // std::string pitch = osspitch.str();

    std::ostringstream ossyaw;
    ossyaw << setiosflags(std::ios::fixed) << std::setprecision(1) << yaw;
    // std::string yaws = ossyaw.str();

    std_msgs::String str_temp;

    str_temp.data = "x: " + xs + "m y: " + ys + "m z: " + zs + "m yaw:" + ossyaw.str() + "°";
    // str_temp.data = "x:  " + xs + "m y: " + ys + "m z: " + zs + "m ";
    // str_temp.data += "roll: " + ossroll.str() +"° pitch: " + osspitch.str() +"° yaw: " + ossyaw.str() + "°";

    string_pub.publish(str_temp);
    dzlog_info("x,y,z,rpy(deg) is :%0.3f , %0.3f , %0.3f , %0.3f , %0.3f , %0.3f , %0.2f. MatchValue is: %f.",stamp.toSec(), odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z , roll, pitch, yaw , MatchValue );

    static std::ofstream ttttttt("/home/map/hdl_pose.txt", std::ios::out);
    static std::ofstream ofs;
    ofs.open("/home/map/hdl_pose.txt", std::ios::app);
    ofs << std::to_string(stamp.toSec()) << " " << std::to_string(odom.pose.pose.position.x) << " " << std::to_string(odom.pose.pose.position.y) << " "
        << std::to_string(odom.pose.pose.position.z) << " " << std::to_string(odom.pose.pose.orientation.x) << " " << std::to_string(odom.pose.pose.orientation.y) << " "
        << std::to_string(odom.pose.pose.orientation.z) << " " << std::to_string(odom.pose.pose.orientation.w) << std::endl;
    ofs.close();
  }

  /**
   * @brief publish scan matching status information
   */
  void publish_scan_matching_status(const std_msgs::Header& header, pcl::PointCloud<pcl::PointXYZI>::Ptr aligned) {
    ScanMatchingStatus status;
    status.header = header;

    status.has_converged = registration->hasConverged();
    status.matching_error = registration->getFitnessScore();
    if (!status.has_converged) 
    {
      dzlog_info("has_converged is %d, matching_error(getFitnessScore) is %f", status.has_converged, status.matching_error);
    }

    double max_correspondence_dist = 0.1;
    max_correspondence_dist = private_nh.param<double>("max_correspondence_dist", 0.1);
    // dzlog_info("max_correspondence_dist is %lf, HDl matching_error(getFitnessScore) is %f", max_correspondence_dist, status.matching_error);

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

    // 匹配好的点<内点比例> 大于 该阈值认为 定位成功
    float loc_success_threshold = 0.90;
    loc_success_threshold = private_nh.param<double>("loc_success_threshold", loc_success_threshold);

    // dzlog_info(" status.inlier_fraction %lf , time %f . aligned->size() %d ", status.inlier_fraction, header.stamp.toSec(), aligned->size());
    static int relocate_cnts = 1;
	  if( relocate )
	  {
	  	publish_odometry(header.stamp, registration->getFinalTransformation(), status.inlier_fraction );
      dzlog_info(" relocate is true ,return hdl pose . time %d . ", relocate_cnts);
	  	if (relocate_cnts++ > 5 )
	  	{
	  		relocate_cnts = 0;
	  		relocate = false;
	  	}		
	  	return;
	  }

    if (status.inlier_fraction > loc_success_threshold) 
    {
      publish_odometry(header.stamp, registration->getFinalTransformation(), status.inlier_fraction );
    }
    else 
    {
      if ( use_odom_flag )
      {
        return;
      }
      if (! private_nh.param<bool>("enable_robot_odometry_prediction", false) ) 
      {
        dzlog_info(" enable_robot_odometry_prediction false .  use hdl result . ");
		    publish_odometry(header.stamp, registration->getFinalTransformation(), status.inlier_fraction );
        return;
      }
      Eigen::Isometry3d PoseDelta = Eigen::Isometry3d::Identity();
      geometry_msgs::TransformStamped odom_delta;

      // 时间戳调转一下
      if (tf_buffer.canTransform(odom_child_frame_id,               last_odom_pose.header.stamp, odom_child_frame_id, header.stamp, robot_odom_frame_id, wait_tf_duration)) {
        odom_delta = tf_buffer.lookupTransform(odom_child_frame_id, last_odom_pose.header.stamp, odom_child_frame_id, header.stamp, robot_odom_frame_id, wait_tf_duration);
        PoseDelta = tf2::transformToEigen(odom_delta);
        // ROS_WARN("    %lf   ", header.stamp.toSec());
        // ROS_WARN_STREAM("pose delta: xyz : "  << std::endl << PoseDelta.matrix() << std::endl);
      } else {
        dzlog_info(" can not interlp with odom ...... ");
	    publish_odometry(header.stamp, registration->getFinalTransformation(), status.inlier_fraction );
        return;
      }

      Eigen::Isometry3d now_pose_eigen = last_pose_eigen * PoseDelta;
      // now_pose_eigen = PoseDelta * last_pose_eigen;
      // std::cout <<  std::endl << last_pose_eigen.matrix() << std::endl << std::endl;
      // std::cout <<  std::endl << now_pose_eigen.matrix() << std::endl << std::endl;

      pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>());
      // 把当前点云 按补偿后的 位姿 投影到地图中，然后查找，看匹配 哪个高一些
      pcl::transformPointCloud(*last_scan, *transformed_cloud, now_pose_eigen.matrix());

      num_inliers = 0;
      for (int i = 0; i < transformed_cloud->size(); i++) {
        const auto& pt = transformed_cloud->at(i);
        registration->getSearchMethodTarget()->nearestKSearch(pt, 1, k_indices, k_sq_dists);
        if (k_sq_dists[0] < max_correspondence_dist * max_correspondence_dist) {
          num_inliers++;
        }
      }
      double compensate_inlier_fraction = static_cast<float>(num_inliers) / transformed_cloud->size();

      // 哪个 匹配度 高 发布哪一个
      // ! 如果 此时的里程计是 准确的，那么 补偿后的 匹配度会更高
      if (compensate_inlier_fraction > status.inlier_fraction) {
        dzlog_info("use odom to get pose . hdl confidence  is  %lf,  compensate_confidence_with_odom %lf ", status.inlier_fraction, compensate_inlier_fraction);
        publish_odometry(header.stamp, now_pose_eigen.matrix().cast<float>(), compensate_inlier_fraction);  // 发布用里程计补偿的位姿
        *aligned = *transformed_cloud;
      } else
        publish_odometry(header.stamp, registration->getFinalTransformation(), status.inlier_fraction);     // 还用老的位姿
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
  ros::Subscriber switchMapSts_sub;

  ros::Publisher pose_pub;
  ros::Publisher aligned_pub;
  ros::Publisher status_pub;
  ros::Publisher confidence_pub;
  ros::Publisher string_pub;
  ros::Publisher current_pose_pub;
  ros::Publisher keda_current_pub;
  ros::Publisher map_name_pub;
  ros::Publisher pub_map_info_status_;
  ros::Timer map_info_pub_timer_;

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

  int iRegionID;
  int iMapID;
  int iNavType;
  int iStatus;
  bool relocate;
  ros::Duration wait_tf_duration;

  // odom input buffer
  nav_msgs::Odometry last_odom_pose;
  Eigen::Isometry3d last_pose_eigen;
  bool use_odom_flag = false;
  float rangeMinX = -250 , rangeMinY = -250 , rangeMinZ = -250 ;
  float rangeMaxX = 250 , rangeMaxY = 250 , rangeMaxZ = 250 ;
};
}  // namespace hdl_localization

PLUGINLIB_EXPORT_CLASS(hdl_localization::HdlLocalizationNodelet, nodelet::Nodelet)
