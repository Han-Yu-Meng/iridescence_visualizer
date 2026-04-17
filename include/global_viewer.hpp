#pragma once

#include <guik/viewer/light_viewer.hpp>
#include <glk/pointcloud_buffer.hpp>
#include <glk/primitives/primitives.hpp>
#include <glk/thin_lines.hpp>
#include <imgui.h> // 引入 ImGui 用于渲染 UI

#include <mutex>
#include <map>
#include <vector>
#include <deque>
#include <chrono>
#include <Eigen/Core>
#include <string>
#include <memory>
#include <atomic>
#include <thread>

class GlobalViewer {
public:
  static GlobalViewer& get() {
    static GlobalViewer instance;
    return instance;
  }

  void register_active_node() {
    if (active_nodes_count_++ == 0) {
      start_viewer();
    }
  }

  void unregister_active_node() {
    if (--active_nodes_count_ == 0) {
      stop_viewer();
    }
  }

  void update_cloud(const std::string& name, const std::vector<float>& buffer) {
    std::lock_guard<std::mutex> lock(mtx_);
    
    // 将新到达的点云及当前时间戳压入接收队列
    incoming_clouds_[name].push_back({buffer, std::chrono::steady_clock::now()});
    
    // 如果是第一次收到该 name 的点云，初始化其 UI 设置参数
    if (cloud_settings_.find(name) == cloud_settings_.end()) {
      cloud_settings_[name] = CloudSettings();
    }
  }

  void update_path(const std::string& name, const std::vector<Eigen::Vector3f>& points) {
    std::lock_guard<std::mutex> lock(mtx_);
    paths_to_update_[name] = points;
  }

  void update_transform(const std::string& name, const Eigen::Matrix4f& matrix, const std::string& child, const std::string& parent) {
    (void)child;
    (void)parent;
    std::lock_guard<std::mutex> lock(mtx_);
    transforms_to_update_[name] = matrix;
  }

private:
  GlobalViewer() : active_nodes_count_(0), running_(false) {}
  ~GlobalViewer() { stop_viewer(); }

  // 为每个点云独立保存的 UI 参数
  struct CloudSettings {
    float point_size = 2.0f;
    float decay_time = 0.0f; // 0.0 表示仅保留最新一帧，>0 表示保留 N 秒内的所有帧
  };

  // 刚接收到的原始数据缓冲
  struct IncomingCloud {
    std::vector<float> buffer;
    std::chrono::steady_clock::time_point timestamp;
  };

  // 已转化为 OpenGL 对象的历史帧
  struct CloudFrame {
    std::string sub_name; // 独一无二的绘制 ID (例如: "PCL Cloud_123")
    std::shared_ptr<glk::PointCloudBuffer> buffer;
    std::chrono::steady_clock::time_point timestamp;
  };

  void start_viewer() {
    if (running_) return;
    running_ = true;
    viewer_thread_ = std::thread(&GlobalViewer::viewer_loop, this);
  }

  void stop_viewer() {
    running_ = false;
    if (viewer_thread_.joinable()) {
      viewer_thread_.join();
    }
  }

  void viewer_loop() {
    // 强制提升支持版本以激活 WSL2 下 RTX 4060 硬件加速
    setenv("MESA_GL_VERSION_OVERRIDE", "4.6", 1);
    setenv("MESA_GL_VERSION_OVERRIDE_FC", "4.6", 1);
    setenv("MESA_GLSL_VERSION_OVERRIDE", "460", 1);
    setenv("MESA_NO_ERROR", "1", 1);
    
    auto viewer = guik::LightViewer::instance();
    if (!viewer) {
      running_ = false;
      return;
    }

    // =========================================================
    // 注册 ImGui UI 面板回调 (此回调会在 viewer->spin_once() 中被触发)
    // =========================================================
    viewer->register_ui_callback("Cloud Controls UI", [&]() {
      std::lock_guard<std::mutex> lock(mtx_); // 锁定资源
      
      // 设置窗口初始大小和位置（可选，增强易用性）
      ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);
      
      ImGui::Begin("Point Cloud Settings");
      
      // 显示渲染 FPS
      ImGui::Text("Render Performance: %.1f FPS", ImGui::GetIO().Framerate);
      ImGui::Separator();
      
      // 增大字体/控件比例 (1.5倍)
      float old_scale = ImGui::GetIO().FontGlobalScale;
      ImGui::GetIO().FontGlobalScale = 1.5f;
      ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.5f);

      for (auto& [name, settings] : cloud_settings_) {
        // 为每一个点云话题创建一个折叠面板
        if (ImGui::CollapsingHeader(name.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
          ImGui::PushID(name.c_str()); // 防止不同组的滑动条 ID 冲突
          
          ImGui::SliderFloat("Point Size", &settings.point_size, 0.1f, 2.0f, "%.1f");
          ImGui::SliderFloat("Decay Time (s)", &settings.decay_time, 0.0f, 600.0f, "%.1f");
          
          ImGui::PopID();
        }
      }
      
      ImGui::GetIO().FontGlobalScale = old_scale; // 恢复缩放，避免影响其他可能的 UI
      ImGui::End();
    });

    uint64_t global_frame_counter = 0; // 用于为不同的帧生成唯一的后缀

    while (running_ && !viewer->closed()) {
      {
        std::lock_guard<std::mutex> lock(mtx_);
        auto now = std::chrono::steady_clock::now();
        
        // ---------------------------------------------------
        // 1. 处理新接收到的点云数据并转换为 OpenGL Buffer
        // ---------------------------------------------------
        for (auto& [name, queue] : incoming_clouds_) {
          for (const auto& incoming : queue) {
            if (incoming.buffer.empty()) {
              // 收到空 buffer，清空该 name 的所有画面
              for (const auto& frame : cloud_histories_[name]) {
                viewer->remove_drawable(frame.sub_name);
              }
              cloud_histories_[name].clear();
              continue;
            }
            
            std::vector<Eigen::Vector3f> points;
            std::vector<Eigen::Vector4f> colors;
            points.reserve(incoming.buffer.size() / 6);
            colors.reserve(incoming.buffer.size() / 6);
            
            for (size_t i = 0; i < incoming.buffer.size(); i += 6) {
              points.emplace_back(incoming.buffer[i], incoming.buffer[i+1], incoming.buffer[i+2]);
              colors.emplace_back(incoming.buffer[i+3], incoming.buffer[i+4], incoming.buffer[i+5], 1.0f);
            }
            
            auto cloud_buffer = std::make_shared<glk::PointCloudBuffer>(points);
            cloud_buffer->add_color(colors);
            
            // 为这一帧生成唯一名称并存入历史队列
            std::string sub_name = name + "_" + std::to_string(global_frame_counter++);
            cloud_histories_[name].push_back({sub_name, cloud_buffer, incoming.timestamp});
          }
          queue.clear(); // 处理完毕清空接收队列
        }

        // ---------------------------------------------------
        // 2. 根据 UI 的衰减设置（Decay Time）更新或销毁点云帧
        // ---------------------------------------------------
        for (auto& [name, history] : cloud_histories_) {
          auto& settings = cloud_settings_[name];

          if (settings.decay_time <= 0.0f) {
            // Decay == 0: 仅保留最新的一帧，旧的全部删掉（经典刷新模式）
            while (history.size() > 1) {
              viewer->remove_drawable(history.front().sub_name);
              history.pop_front();
            }
          } else {
            // Decay > 0: 判断超时，清理过期的历史帧
            while (!history.empty()) {
              std::chrono::duration<float> age = now - history.front().timestamp;
              if (age.count() > settings.decay_time) {
                viewer->remove_drawable(history.front().sub_name);
                history.pop_front();
              } else {
                break; // 由于双端队列是按时间顺序插入的，第一个没超时，后面的肯定也没超时
              }
            }
          }

          // ---------------------------------------------------
          // 3. 将现存有效帧发送给 Viewer，并动态应用点大小
          // ---------------------------------------------------
          for (const auto& frame : history) {
            viewer->update_drawable(frame.sub_name, frame.buffer, 
                                    guik::VertexColor().set_point_scale(settings.point_size));
          }
        }

        // ---------------------------------------------------
        // 4. 更新轨迹 (Path) 和 坐标系 Transform 
        // ---------------------------------------------------
        for (auto& [name, points] : paths_to_update_) {
          if (points.empty()) {
            viewer->remove_drawable(name);
          } else {
            viewer->update_thin_lines(name, points, true, guik::FlatOrange());
          }
        }
        paths_to_update_.clear();

        for (auto& [name, matrix] : transforms_to_update_) {
          viewer->update_drawable(name, glk::Primitives::coordinate_system(), guik::VertexColor().set_model_matrix(matrix));
        }
        transforms_to_update_.clear();
      } // <--- std::lock_guard 作用域结束释放锁，防止下面的 spin_once 死锁

      viewer->spin_once();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    running_ = false;
  }

  std::atomic<int> active_nodes_count_;
  std::atomic<bool> running_;
  std::thread viewer_thread_;
  std::mutex mtx_;

  // UI 参数存储与点云历史队列
  std::map<std::string, CloudSettings> cloud_settings_;
  std::map<std::string, std::vector<IncomingCloud>> incoming_clouds_;
  std::map<std::string, std::deque<CloudFrame>> cloud_histories_;

  std::map<std::string, std::vector<Eigen::Vector3f>> paths_to_update_;
  std::map<std::string, Eigen::Matrix4f> transforms_to_update_;
};