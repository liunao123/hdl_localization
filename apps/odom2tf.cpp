#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>

static std::string topic = "/Odometry";
static std::string parent_frame = "odom";
static std::string child_frame = "base_link";

void odom_callback(const nav_msgs::OdometryConstPtr& odom){
  static tf::TransformBroadcaster br;
  tf::Transform tf;
  geometry_msgs::Pose odom_pose = odom->pose.pose;

//  tf.setOrigin(tf::Vector3(odom_pose.position.x, odom_pose.position.y, odom_pose.position.z));
//  tf::Quaternion quat;
//  tf::quaternionMsgToTF(odom_pose.orientation, quat);
//  quat.normalize();
//  tf.setRotation(quat);
  tf::poseMsgToTF(odom_pose, tf);

  tf::StampedTransform stamped_tf(tf, odom->header.stamp, parent_frame, child_frame);
  // auto st = ros::Time::now();
  // tf::StampedTransform stamped_tf(tf, st, parent_frame, child_frame);
  // ROS_WARN("tf time %lf ", st.toSec() );

  br.sendTransform(stamped_tf);
}




int main(int argc, char **argv){

  ros::init(argc, argv, "odom2tf");
  ros::NodeHandle nh;
  ros::NodeHandle private_nh("~");

  private_nh.getParam("odom_topic", topic);
  private_nh.getParam("parent_frame", parent_frame);
  private_nh.getParam("child_frame", child_frame);

  std::cout << "Topic: " << topic << std::endl;
  std::cout << "Parent frame: " << parent_frame << std::endl;
  std::cout << "Child frame: " << child_frame << std::endl;

  ros::Subscriber odom_sub = nh.subscribe(topic, 10, odom_callback);
  ros::spin();

  return 0;
}