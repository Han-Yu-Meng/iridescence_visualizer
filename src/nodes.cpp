#include <fins/node.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <Eigen/Geometry>

#include "global_viewer.hpp"

inline void intensity_to_rainbow(float value, float min_v, float max_v,
                                 float &r, float &g, float &b) {
  float h = (1.0f - (value - min_v) / (max_v - min_v)) * 240.0f;
  if (h < 0) h = 0;
  if (h > 240) h = 240;
  float x = 1.0f - std::abs(std::fmod(h / 60.0f, 2.0f) - 1.0f);
  if (h < 60) { r = 1; g = x; b = 0; }
  else if (h < 120) { r = x; g = 1; b = 0; }
  else if (h < 180) { r = 0; g = 1; b = x; }
  else if (h < 240) { r = 0; g = x; b = 1; }
  else { r = x; g = 0; b = 1; }
}

class BaseIridescenceVisualizer : public fins::Node {
protected:
  std::string title_;

public:
  void define() override {
    register_parameter<std::string>("title", &BaseIridescenceVisualizer::update_title, "PCL Cloud");
  }
  void update_title(const std::string &new_title) { title_ = new_title; }
  
  void run() override { GlobalViewer::get().register_active_node(); }
  void pause() override { GlobalViewer::get().unregister_active_node(); }
  void reset() override { GlobalViewer::get().unregister_active_node(); }
  void initialize() override {}
};

class IridescenceCloudXYZVisualizer : public BaseIridescenceVisualizer {
public:
  void define() override {
    BaseIridescenceVisualizer::define();
    set_name("IridescenceCloudXYZVisualizer");
    set_description("Visualizes XYZ PointCloud in Iridescence");
    set_category("Visualization>Iridescence");
    register_input<pcl::PointCloud<pcl::PointXYZ>::Ptr>("cloud", &IridescenceCloudXYZVisualizer::on_cloud);
  }

  void on_cloud(const fins::Msg<pcl::PointCloud<pcl::PointXYZ>::Ptr> &msg) {
    auto cloud = *msg;
    if (!cloud || cloud->empty()) return;
    std::vector<float> buffer;
    buffer.reserve(cloud->size() * 6);
    for (const auto &p : *cloud) {
      buffer.push_back(p.x); buffer.push_back(p.y); buffer.push_back(p.z);
      buffer.push_back(0.8f); buffer.push_back(0.8f); buffer.push_back(0.8f); // Grey
    }
    GlobalViewer::get().update_cloud(title_, buffer);
  }
};

class IridescenceCloudXYZIVisualizer : public BaseIridescenceVisualizer {
public:
  void define() override {
    BaseIridescenceVisualizer::define();
    set_name("IridescenceCloudXYZIVisualizer");
    set_description("Visualizes XYZI PointCloud with Intensity");
    set_category("Visualization>Iridescence");
    register_input<pcl::PointCloud<pcl::PointXYZI>::Ptr>("cloud", &IridescenceCloudXYZIVisualizer::on_cloud);
  }

  void on_cloud(const fins::Msg<pcl::PointCloud<pcl::PointXYZI>::Ptr> &msg) {
    auto cloud = *msg;
    if (!cloud || cloud->empty()) return;
    
    float min_i = 1e10, max_i = -1e10;
    for (const auto &p : *cloud) {
      min_i = std::min(min_i, p.intensity);
      max_i = std::max(max_i, p.intensity);
    }
    if (max_i <= min_i) max_i = min_i + 1.0f;

    std::vector<float> buffer;
    buffer.reserve(cloud->size() * 6);
    for (const auto &p : *cloud) {
      buffer.push_back(p.x); buffer.push_back(p.y); buffer.push_back(p.z);
      float r, g, b;
      intensity_to_rainbow(p.intensity, min_i, max_i, r, g, b);
      buffer.push_back(r); buffer.push_back(g); buffer.push_back(b);
    }
    GlobalViewer::get().update_cloud(title_, buffer);
  }
};

class IridescenceCloudRGBVisualizer : public BaseIridescenceVisualizer {
public:
  void define() override {
    BaseIridescenceVisualizer::define();
    set_name("IridescenceCloudRGBVisualizer");
    set_description("Visualizes XYZRGB PointCloud");
    set_category("Visualization>Iridescence");
    register_input<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>("cloud", &IridescenceCloudRGBVisualizer::on_cloud);
  }

  void on_cloud(const fins::Msg<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &msg) {
    auto cloud = *msg;
    if (!cloud || cloud->empty()) return;
    std::vector<float> buffer;
    buffer.reserve(cloud->size() * 6);
    for (const auto &p : *cloud) {
      buffer.push_back(p.x); buffer.push_back(p.y); buffer.push_back(p.z);
      buffer.push_back(p.r / 255.0f); buffer.push_back(p.g / 255.0f); buffer.push_back(p.b / 255.0f);
    }
    GlobalViewer::get().update_cloud(title_, buffer);
  }
};

class IridescenceTransformVisualizer : public fins::Node {
  std::string title_;
public:
  void define() override {
    set_name("IridescenceTransformVisualizer");
    set_category("Visualization>Iridescence");
    register_parameter<std::string>("title", &IridescenceTransformVisualizer::update_title, "Tf");
    register_input<geometry_msgs::msg::TransformStamped>("transform", &IridescenceTransformVisualizer::on_transform);
  }
  void update_title(const std::string &t) { title_ = t; }
  void run() override { GlobalViewer::get().register_active_node(); }
  void pause() override { GlobalViewer::get().unregister_active_node(); }
  void reset() override { GlobalViewer::get().unregister_active_node(); }
  void initialize() override {}
  
  void on_transform(const fins::Msg<geometry_msgs::msg::TransformStamped> &msg) {
    Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
    mat(0,3) = msg->transform.translation.x;
    mat(1,3) = msg->transform.translation.y;
    mat(2,3) = msg->transform.translation.z;
    Eigen::Quaternionf q(msg->transform.rotation.w, msg->transform.rotation.x, msg->transform.rotation.y, msg->transform.rotation.z);
    mat.block<3,3>(0,0) = q.toRotationMatrix();
    GlobalViewer::get().update_transform(title_, mat, msg->child_frame_id, msg->header.frame_id);
  }
};

class IridescencePathVisualizer : public fins::Node {
  std::string title_;
public:
  void define() override {
    set_name("IridescencePathVisualizer");
    set_category("Visualization>Iridescence");
    register_parameter<std::string>("title", &IridescencePathVisualizer::update_title, "Path");
    register_input<nav_msgs::msg::Path>("path", &IridescencePathVisualizer::on_path);
  }
  void update_title(const std::string &t) { title_ = t; }
  void run() override { GlobalViewer::get().register_active_node(); }
  void pause() override { GlobalViewer::get().unregister_active_node(); }
  void reset() override { GlobalViewer::get().unregister_active_node(); }
  void initialize() override {}

  void on_path(const fins::Msg<nav_msgs::msg::Path> &msg) {
    std::vector<Eigen::Vector3f> points;
    for (const auto &p : msg->poses) {
      points.emplace_back(p.pose.position.x, p.pose.position.y, p.pose.position.z);
    }
    GlobalViewer::get().update_path(title_, points);
  }
};

EXPORT_NODE(IridescenceCloudXYZVisualizer)
EXPORT_NODE(IridescenceCloudXYZIVisualizer)
EXPORT_NODE(IridescenceCloudRGBVisualizer)
EXPORT_NODE(IridescenceTransformVisualizer)
EXPORT_NODE(IridescencePathVisualizer)
DEFINE_PLUGIN_ENTRY()
