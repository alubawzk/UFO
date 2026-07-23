// Minimal XRoboToolkit binding required by pico_sim2sim/sonic_server.py.
// Derived from XRoboToolkit-PC-Service-Pybind (MIT); see third_party/.

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "PXREARobotSDK.h"

namespace py = pybind11;
using json = nlohmann::json;

namespace {

std::array<std::array<double, 7>, 24> body_joints_pose{};
std::int64_t timestamp_ns = 0;
bool body_data_available = false;
std::mutex state_mutex;

std::array<double, 7> parse_pose(const std::string& text) {
  std::array<double, 7> result{};
  std::stringstream stream(text);
  std::string value;
  std::size_t index = 0;
  while (std::getline(stream, value, ',') && index < result.size()) {
    result[index++] = std::stod(value);
  }
  return result;
}

void callback(void*, PXREAClientCallbackType type, int, void* user_data) {
  if (type == PXREAServerConnect) {
    std::cout << "[xrobotoolkit] server connected" << std::endl;
    return;
  }
  if (type == PXREAServerDisconnect) {
    std::cout << "[xrobotoolkit] server disconnected" << std::endl;
    return;
  }
  if (type != PXREADeviceStateJson || user_data == nullptr) {
    return;
  }

  try {
    const auto& device_state = *static_cast<PXREADevStateJson*>(user_data);
    const json envelope = json::parse(device_state.stateJson);
    if (!envelope.contains("value")) {
      return;
    }
    const json value = json::parse(envelope.at("value").get<std::string>());
    std::lock_guard<std::mutex> lock(state_mutex);
    if (value.contains("timeStampNs")) {
      timestamp_ns = value.at("timeStampNs").get<std::int64_t>();
    }
    if (!value.contains("Body")) {
      return;
    }
    const json& body = value.at("Body");
    if (body.contains("timeStampNs")) {
      timestamp_ns = body.at("timeStampNs").get<std::int64_t>();
    }
    if (!body.contains("joints") || !body.at("joints").is_array()) {
      return;
    }
    const json& joints = body.at("joints");
    const std::size_t count = std::min<std::size_t>(joints.size(), body_joints_pose.size());
    for (std::size_t index = 0; index < count; ++index) {
      if (joints.at(index).contains("p")) {
        body_joints_pose[index] = parse_pose(joints.at(index).at("p").get<std::string>());
      }
    }
    body_data_available = count == body_joints_pose.size();
  } catch (const std::exception& exception) {
    std::cerr << "[xrobotoolkit] state parse error: " << exception.what() << std::endl;
  }
}

void init() {
  if (PXREAInit(nullptr, callback, PXREAFullMask) != 0) {
    throw std::runtime_error("PXREAInit failed");
  }
}

void deinit() {
  PXREADeinit();
}

bool is_body_data_available() {
  std::lock_guard<std::mutex> lock(state_mutex);
  return body_data_available;
}

std::int64_t get_time_stamp_ns() {
  std::lock_guard<std::mutex> lock(state_mutex);
  return timestamp_ns;
}

std::array<std::array<double, 7>, 24> get_body_joints_pose() {
  std::lock_guard<std::mutex> lock(state_mutex);
  return body_joints_pose;
}

}  // namespace

PYBIND11_MODULE(xrobotoolkit_sdk, module) {
  module.doc() = "Minimal XRoboToolkit body-tracking binding for UFO pico_sim2sim";
  module.def("init", &init);
  module.def("close", &deinit);
  module.def("is_body_data_available", &is_body_data_available);
  module.def("get_time_stamp_ns", &get_time_stamp_ns);
  module.def("get_body_joints_pose", &get_body_joints_pose);
}
